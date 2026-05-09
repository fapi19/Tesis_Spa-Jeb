"""Rare-token / morphology-aware evaluation (Enhancement #2).

Beyond the corpus-level BLEU/chrF++ that Phase 5 reports, low-resource
agglutinative languages benefit from explicit reporting of how a system
behaves on rare and out-of-vocabulary forms. We bucket the test set by the
fraction of *rare* words in the reference (frequency in train below a
threshold) and report chrF++ for each bucket. We also report an
``oov_recovery_rate`` that measures how often words in the reference that
were never seen in train (true OOVs) are also produced verbatim in the
hypothesis.

The headline aggregate metric ``rare_token_chrf`` corresponds to the
``>=20% rare`` bucket, which is the cleanest single-number proxy for
"how well does the system handle the morphologically dense tail".
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .metrics import MetricsConfig, compute_bleu_chrf

_WORD_RE = re.compile(r"[^\s]+")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip().lower()


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization on NFC-lowercased text.

    Apostrophes are preserved (critical for Shiwilu morphology). Punctuation
    glued to a word stays attached; this matches how chrF++ counts characters
    and how the train CSV looks.
    """
    return _WORD_RE.findall(_normalize(text))


def compute_train_freqs(train_csv: Path) -> dict[str, Counter[str]]:
    """Per-language word frequency over the training set.

    Returns a dict ``{"shw": Counter, "spa": Counter}``. The CSV is the
    canonical bidirectional table emitted by Phase 1; we count over the
    ``target`` column grouped by ``target_lang`` so that each row contributes
    its own language exactly once (avoids double counting because each pair
    appears in both directions).
    """
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    if "target" not in df.columns or "target_lang" not in df.columns:
        raise ValueError(
            f"{train_csv} is missing 'target'/'target_lang' columns; got {list(df.columns)}"
        )
    out: dict[str, Counter[str]] = {}
    for lang, sub in df.groupby("target_lang"):
        counter: Counter[str] = Counter()
        for text in sub["target"].astype(str):
            counter.update(_tokenize(text))
        out[str(lang)] = counter
    return out


@dataclass(frozen=True)
class BucketSpec:
    name: str
    lower: float
    upper: float

    def contains(self, ratio: float) -> bool:
        return self.lower <= ratio < self.upper


DEFAULT_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec("0pct", 0.0, 1e-9),
    BucketSpec("0_to_20pct", 1e-9, 0.2),
    BucketSpec("20_to_50pct", 0.2, 0.5),
    BucketSpec("50pct_or_more", 0.5, 1.000001),
)


def _ratio(text: str, freqs: Counter[str], rare_threshold: int) -> tuple[int, int]:
    """Return ``(rare_count, total_count)`` for a single reference."""
    tokens = _tokenize(text)
    if not tokens:
        return (0, 0)
    rare = sum(1 for tok in tokens if freqs.get(tok, 0) < rare_threshold)
    return rare, len(tokens)


def _oov_recovery(reference: str, hypothesis: str, vocab: set[str]) -> tuple[int, int]:
    """Count true-OOV reference words that survive into the hypothesis verbatim."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = set(_tokenize(hypothesis))
    oov_in_ref = [tok for tok in ref_tokens if tok not in vocab]
    if not oov_in_ref:
        return (0, 0)
    recovered = sum(1 for tok in oov_in_ref if tok in hyp_tokens)
    return recovered, len(oov_in_ref)


def _safe_chrf(
    hypotheses: list[str],
    references: list[str],
    cfg: MetricsConfig,
) -> dict[str, float]:
    if not hypotheses:
        return {"bleu": float("nan"), "chrf_pp": float("nan")}
    metrics = compute_bleu_chrf(hypotheses, references, cfg)
    return {"bleu": metrics["bleu"], "chrf_pp": metrics["chrf_pp"]}


@dataclass(frozen=True)
class RareTokenConfig:
    rare_threshold: int = 5
    buckets: tuple[BucketSpec, ...] = DEFAULT_BUCKETS
    headline_bucket: str = "20pct_or_more"


def _direction_target_lang(direction: str) -> str:
    """Return target language code from a Phase 5-style direction string."""
    if "2" not in direction:
        raise ValueError(f"unexpected direction format: {direction!r}")
    return direction.split("2", 1)[1]


def evaluate_rare_tokens(
    predictions: list[dict[str, Any]],
    train_freqs: dict[str, Counter[str]],
    metrics_cfg: MetricsConfig,
    rare_cfg: RareTokenConfig | None = None,
) -> dict[str, Any]:
    """Run the bucketed rare-token analysis on a list of predictions.

    Parameters
    ----------
    predictions:
        Same dict shape Phase 5 emits: ``id, direction, source, reference,
        hypothesis``. The optional ``origin_source`` field, if present, is
        used to produce a per-source breakdown.
    train_freqs:
        Output of :func:`compute_train_freqs` over the training table.
    metrics_cfg:
        Phase 5 ``MetricsConfig`` (chrF++ + BLEU). We deliberately avoid
        BERTScore/COMET here: bucket sizes are small, so per-bucket COMET
        would be noisy; chrF++ remains the headline.
    """
    rare_cfg = rare_cfg or RareTokenConfig()

    by_direction: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        by_direction.setdefault(p["direction"], []).append(p)

    out: dict[str, Any] = {
        "config": {
            "rare_threshold": rare_cfg.rare_threshold,
            "buckets": [{"name": b.name, "lower": b.lower, "upper": b.upper} for b in rare_cfg.buckets],
            "headline_bucket": rare_cfg.headline_bucket,
        },
        "directions": {},
    }

    for direction, items in by_direction.items():
        target_lang = _direction_target_lang(direction)
        freqs = train_freqs.get(target_lang, Counter())
        vocab = set(freqs.keys())

        tagged: list[dict[str, Any]] = []
        for p in items:
            rare_count, total = _ratio(p["reference"], freqs, rare_cfg.rare_threshold)
            ratio = rare_count / total if total else 0.0
            recovered, oov_total = _oov_recovery(p["reference"], p["hypothesis"], vocab)
            tagged.append(
                {
                    **p,
                    "rare_word_count": rare_count,
                    "rare_word_total": total,
                    "rare_word_ratio": ratio,
                    "oov_recovered": recovered,
                    "oov_total": oov_total,
                }
            )

        bucket_stats: dict[str, Any] = {}
        for spec in rare_cfg.buckets:
            sub = [t for t in tagged if spec.contains(t["rare_word_ratio"])]
            sub_metrics = _safe_chrf(
                [t["hypothesis"] for t in sub],
                [t["reference"] for t in sub],
                metrics_cfg,
            )
            bucket_stats[spec.name] = {
                "n": len(sub),
                "mean_rare_ratio": float(sum(t["rare_word_ratio"] for t in sub) / len(sub)) if sub else float("nan"),
                **sub_metrics,
            }

        # Headline bucket: combine 20-50pct and >=50pct since both are "rare-heavy".
        rare_heavy = [t for t in tagged if t["rare_word_ratio"] >= 0.2]
        rare_heavy_metrics = _safe_chrf(
            [t["hypothesis"] for t in rare_heavy],
            [t["reference"] for t in rare_heavy],
            metrics_cfg,
        )
        bucket_stats["20pct_or_more"] = {
            "n": len(rare_heavy),
            "mean_rare_ratio": (
                float(sum(t["rare_word_ratio"] for t in rare_heavy) / len(rare_heavy))
                if rare_heavy
                else float("nan")
            ),
            **rare_heavy_metrics,
        }

        oov_recovered_total = sum(t["oov_recovered"] for t in tagged)
        oov_total = sum(t["oov_total"] for t in tagged)
        oov_recovery_rate = float(oov_recovered_total / oov_total) if oov_total else float("nan")

        per_origin: dict[str, dict[str, Any]] = {}
        if any("origin_source" in t for t in tagged):
            for origin, sub in _group_by(tagged, key="origin_source").items():
                per_origin[str(origin)] = {
                    "n": len(sub),
                    **_safe_chrf(
                        [t["hypothesis"] for t in sub],
                        [t["reference"] for t in sub],
                        metrics_cfg,
                    ),
                }

        out["directions"][direction] = {
            "target_lang": target_lang,
            "n": len(items),
            "rare_threshold": rare_cfg.rare_threshold,
            "buckets": bucket_stats,
            "oov": {
                "recovered": oov_recovered_total,
                "total": oov_total,
                "recovery_rate": oov_recovery_rate,
            },
            "by_origin_source": per_origin,
        }

    if all(d in out["directions"] for d in ("shw2spa", "spa2shw")):
        avg_rare_chrf = sum(
            out["directions"][d]["buckets"]["20pct_or_more"]["chrf_pp"]
            for d in ("shw2spa", "spa2shw")
            if not _is_nan(out["directions"][d]["buckets"]["20pct_or_more"]["chrf_pp"])
        ) / 2.0
        out["avg_rare_chrf_pp_20pct_or_more"] = avg_rare_chrf
        out["avg_oov_recovery_rate"] = sum(
            out["directions"][d]["oov"]["recovery_rate"]
            for d in ("shw2spa", "spa2shw")
            if not _is_nan(out["directions"][d]["oov"]["recovery_rate"])
        ) / 2.0

    return out


def _is_nan(x: float) -> bool:
    return x != x


def _group_by(items: Iterable[dict[str, Any]], *, key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        k = item.get(key)
        if k is None:
            continue
        out.setdefault(str(k), []).append(item)
    return out


def attach_origin_source(
    predictions: list[dict[str, Any]],
    test_csv: Path,
) -> list[dict[str, Any]]:
    """Hydrate `origin_source` onto predictions from the test CSV.

    Phase 5 strips the field when emitting predictions; we re-attach it here
    via ``id`` for the per-source breakdown.
    """
    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    if "id" not in df.columns or "origin_source" not in df.columns:
        return predictions
    by_id = dict(zip(df["id"].astype(str), df["origin_source"].astype(str)))
    for p in predictions:
        origin = by_id.get(str(p.get("id", "")))
        if origin is not None:
            p["origin_source"] = origin
    return predictions
