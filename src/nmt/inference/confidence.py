"""Reliability / confidence layer for NMT outputs (Enhancement #6).

Two regimes share the same machinery:

- Baseline (no reranker): ``confidence_score = exp(top-1 sequence_score)``,
  i.e. the geometric mean per-token probability assigned by NLLB to its
  best hypothesis. Default thresholds (0.40, 0.55) were calibrated on the
  v0 test outputs (642 rows, beam=5; observed range ~[0.20, 0.79], median
  0.41).
- Reranked: ``confidence_score = final_score = alpha * trans_prob +
  (1 - alpha) * cos_sim`` (already in [0, 1]). Default thresholds
  (0.30, 0.40) were calibrated on the v0+rerank(alpha=0.7) outputs
  (observed range ~[0.10, 0.48], median 0.28). They produce a meaningful
  spread for alpha in [0.5, 0.7]; alpha=0.3 will skew toward high.

Higher band means more reliable, but this is a soft signal, not a guarantee.
The numbers are exposed alongside each prediction so downstream consumers can
apply their own policy (e.g. pre-fill flashcards only with ``high``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import math

import numpy as np

ConfidenceBand = Literal["low", "medium", "high"]

BASELINE_THRESHOLDS: tuple[float, float] = (0.40, 0.55)
RERANKED_THRESHOLDS: tuple[float, float] = (0.30, 0.40)


@dataclass(frozen=True)
class ConfidenceConfig:
    """Two thresholds split a score into three bands.

    A score ``s`` is mapped to:
    - ``"low"``    if ``s < low_to_med``
    - ``"medium"`` if ``low_to_med <= s < med_to_high``
    - ``"high"``   if ``s >= med_to_high``
    """

    low_to_med: float
    med_to_high: float
    name: str

    def __post_init__(self) -> None:
        if self.low_to_med > self.med_to_high:
            raise ValueError(
                f"low_to_med ({self.low_to_med}) must be <= med_to_high ({self.med_to_high})"
            )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "low_to_med": self.low_to_med,
            "med_to_high": self.med_to_high,
        }


def baseline_config() -> ConfidenceConfig:
    """Defaults for ``exp(top-1 sequence_score)`` (geometric mean per-token prob)."""
    return ConfidenceConfig(
        low_to_med=BASELINE_THRESHOLDS[0],
        med_to_high=BASELINE_THRESHOLDS[1],
        name="baseline_exp_seq_score",
    )


def reranked_config(alpha: float | None = None) -> ConfidenceConfig:
    """Defaults for reranked `final_score = alpha*trans_prob + (1-alpha)*cos_sim`."""
    suffix = f"_alpha_{alpha:.2f}" if alpha is not None else ""
    return ConfidenceConfig(
        low_to_med=RERANKED_THRESHOLDS[0],
        med_to_high=RERANKED_THRESHOLDS[1],
        name=f"reranked{suffix}",
    )


def to_band(score: float, cfg: ConfidenceConfig) -> ConfidenceBand:
    if score >= cfg.med_to_high:
        return "high"
    if score >= cfg.low_to_med:
        return "medium"
    return "low"


def softmax_over_nbest(seq_scores: Sequence[float]) -> list[float]:
    """Stable softmax over a single n-best list."""
    if not seq_scores:
        return []
    arr = np.asarray(seq_scores, dtype=np.float64)
    arr = arr - arr.max()
    ex = np.exp(arr)
    out = ex / ex.sum()
    return [float(x) for x in out]


def summarize_bands(
    predictions: Sequence[dict],
    *,
    band_key: str = "confidence",
    direction_key: str = "direction",
) -> dict[str, dict[str, int] | int]:
    """Aggregate band counts overall and per direction.

    Returns a dict with ``"overall"`` -> ``{low, medium, high}`` and one entry
    per direction observed (e.g. ``"shw2spa"``, ``"spa2shw"``).
    """
    overall: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    per_dir: dict[str, dict[str, int]] = {}
    for p in predictions:
        band = p.get(band_key)
        if band not in overall:
            continue
        overall[band] += 1
        direction = p.get(direction_key)
        if direction is None:
            continue
        bucket = per_dir.setdefault(direction, {"low": 0, "medium": 0, "high": 0})
        bucket[band] += 1
    out: dict[str, dict[str, int] | int] = {"overall": overall}
    for d, counts in per_dir.items():
        out[d] = counts
    return out


def attach_baseline_confidence(predictions: list[dict]) -> tuple[list[dict], ConfidenceConfig]:
    """Add `confidence`, `confidence_score`, `confidence_components` to each row.

    Uses ``exp(top-1 sequence_score)`` as the score. The softmax-over-n-best
    top-1 probability is also reported in ``confidence_components`` so the
    JSONL still carries both signals (the reranker uses the second one).
    """
    cfg = baseline_config()
    for p in predictions:
        top_seq_score = float(p.get("sequence_score", 0.0))
        score = float(math.exp(top_seq_score))
        cands = p.get("candidates") or []
        softmax_top1 = 0.0
        if cands:
            seq_scores = [float(c["sequence_score"]) for c in cands]
            probs = softmax_over_nbest(seq_scores)
            softmax_top1 = probs[0]
        p["confidence_score"] = score
        p["confidence"] = to_band(score, cfg)
        p["confidence_components"] = {
            "exp_seq_score": score,
            "softmax_over_nbest_top1": softmax_top1,
            "raw_sequence_score": top_seq_score,
        }
    return predictions, cfg


def attach_reranked_confidence(
    predictions: list[dict],
    *,
    alpha: float,
) -> tuple[list[dict], ConfidenceConfig]:
    """Add `confidence` to each reranked prediction.

    Each row is expected to carry `chosen_rank` and have `final_score`,
    `trans_prob`, `cos_sim` on `candidates[chosen_rank]`.
    """
    cfg = reranked_config(alpha)
    for p in predictions:
        idx = int(p.get("chosen_rank", 0))
        cands = p.get("candidates") or []
        if not cands or idx >= len(cands):
            p["confidence_score"] = 0.0
            p["confidence"] = "low"
            p["confidence_components"] = {
                "trans_prob": 0.0,
                "cos_sim": 0.0,
                "alpha": alpha,
            }
            continue
        chosen = cands[idx]
        final_score = float(chosen.get("final_score", 0.0))
        p["confidence_score"] = final_score
        p["confidence"] = to_band(final_score, cfg)
        p["confidence_components"] = {
            "trans_prob": float(chosen.get("trans_prob", 0.0)),
            "cos_sim": float(chosen.get("cos_sim", 0.0)),
            "alpha": alpha,
        }
    return predictions, cfg
