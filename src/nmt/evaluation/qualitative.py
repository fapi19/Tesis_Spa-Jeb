"""Stratified qualitative analysis of NMT outputs (expert-feedback enhancement).

The corpus-level metrics that Phase 5/6 report hide *where* a system succeeds or
fails. Following the MT-evaluation expert's advice, this module buckets the test
predictions of a single run by a *per-sentence* score and exposes representative
examples per bucket, so the thesis can show qualitatively what each score band
looks like.

Two design choices follow the expert's reasoning directly:

* **Metric per direction.** ``shw2spa`` (Spanish output) is scored with
  sentence-BLEU, because BLEU is informative when the target is a high-resource,
  non-agglutinative language. ``spa2shw`` (Shiwilu output) is scored with
  sentence-chrF++, because BLEU over-penalizes the malformed *whole words* that
  an agglutinative target produces, masking real differences.
* **Noise-floor cut points.** BLEU buckets split at 10 and 20; chrF++ buckets at
  20 and 40. The expert's "ojo de experto" is that BLEU <= 10 (chrF++ <= 20) is
  mostly noise — correct tokens tend to be frequent/short words (pronouns,
  prepositions, punctuation) rather than real content.

The reusable ``BucketSpec`` from :mod:`rare_token` is shared so both analyses
keep the same bucketing semantics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from .metrics import MetricsConfig, sentence_bleu, sentence_chrf_pp
from .rare_token import BucketSpec

# Per-direction sentence-BLEU buckets (Spanish output: shw2spa).
BLEU_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec("bleu_0_10", 0.0, 10.0),
    BucketSpec("bleu_10_20", 10.0, 20.0),
    BucketSpec("bleu_20_plus", 20.0, 1e9),
)

# Per-direction sentence-chrF++ buckets (Shiwilu output: spa2shw).
CHRF_BUCKETS: tuple[BucketSpec, ...] = (
    BucketSpec("chrf_0_20", 0.0, 20.0),
    BucketSpec("chrf_20_40", 20.0, 40.0),
    BucketSpec("chrf_40_plus", 40.0, 1e9),
)

# Which sentence-level metric scores each direction. Defaults to chrF++ for any
# direction not listed (safe for the morphology-rich side).
DIRECTION_METRIC: dict[str, str] = {
    "shw2spa": "bleu",
    "spa2shw": "chrf_pp",
}


@dataclass(frozen=True)
class QualitativeConfig:
    samples_per_bucket: int = 5
    seed: int = 2026
    max_chars: int = 400  # truncate very long texts in the sampled output


def _truncate(text: str, max_chars: int) -> str:
    s = str(text)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _metric_for_direction(
    direction: str,
) -> tuple[str, tuple[BucketSpec, ...], Callable[[str, str, MetricsConfig], float]]:
    metric_name = DIRECTION_METRIC.get(direction, "chrf_pp")
    if metric_name == "bleu":
        return "bleu", BLEU_BUCKETS, sentence_bleu
    return "chrf_pp", CHRF_BUCKETS, sentence_chrf_pp


def _bucket_for(score: float, buckets: tuple[BucketSpec, ...]) -> str:
    for spec in buckets:
        if spec.contains(score):
            return spec.name
    return buckets[-1].name


def evaluate_qualitative(
    predictions: list[dict[str, Any]],
    metrics_cfg: MetricsConfig,
    qual_cfg: QualitativeConfig | None = None,
) -> dict[str, Any]:
    """Bucket predictions by per-sentence score and sample examples per bucket.

    ``predictions`` is the Phase 5/6 dict shape: ``id, direction, source,
    reference, hypothesis`` (``origin_source`` optional). Returns a JSON-able
    dict with, per direction: the scoring metric, bucket counts + mean score,
    and a seeded stratified sample of examples per bucket.
    """
    qual_cfg = qual_cfg or QualitativeConfig()

    by_direction: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        by_direction.setdefault(p["direction"], []).append(p)

    out: dict[str, Any] = {
        "config": {
            "samples_per_bucket": qual_cfg.samples_per_bucket,
            "seed": qual_cfg.seed,
            "direction_metric": dict(DIRECTION_METRIC),
        },
        "directions": {},
    }

    for direction, items in sorted(by_direction.items()):
        metric_name, buckets, score_fn = _metric_for_direction(direction)
        rng = random.Random(f"{qual_cfg.seed}:{direction}")

        tagged: list[dict[str, Any]] = []
        for p in items:
            score = score_fn(p["hypothesis"], p["reference"], metrics_cfg)
            tagged.append({**p, "sentence_score": score, "bucket": _bucket_for(score, buckets)})

        bucket_stats: dict[str, Any] = {}
        samples: dict[str, list[dict[str, Any]]] = {}
        for spec in buckets:
            sub = [t for t in tagged if t["bucket"] == spec.name]
            scores = [t["sentence_score"] for t in sub]
            bucket_stats[spec.name] = {
                "n": len(sub),
                "lower": spec.lower,
                "upper": spec.upper if spec.upper < 1e8 else None,
                "pct": float(len(sub) / len(tagged)) if tagged else float("nan"),
                "mean_score": float(sum(scores) / len(scores)) if scores else float("nan"),
            }
            picked = sub if len(sub) <= qual_cfg.samples_per_bucket else rng.sample(
                sub, qual_cfg.samples_per_bucket
            )
            picked = sorted(picked, key=lambda t: t["sentence_score"])
            samples[spec.name] = [
                {
                    "id": t.get("id"),
                    "pair_id": t.get("pair_id"),
                    "origin_source": t.get("origin_source"),
                    "sentence_score": round(float(t["sentence_score"]), 2),
                    "source": _truncate(t["source"], qual_cfg.max_chars),
                    "reference": _truncate(t["reference"], qual_cfg.max_chars),
                    "hypothesis": _truncate(t["hypothesis"], qual_cfg.max_chars),
                }
                for t in picked
            ]

        out["directions"][direction] = {
            "metric": metric_name,
            "n": len(items),
            "mean_score": float(sum(t["sentence_score"] for t in tagged) / len(tagged))
            if tagged
            else float("nan"),
            "buckets": bucket_stats,
            "samples": samples,
        }

    return out


def sampled_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the per-bucket samples into CSV-ready rows."""
    rows: list[dict[str, Any]] = []
    for direction, body in report["directions"].items():
        for bucket_name, examples in body["samples"].items():
            for ex in examples:
                rows.append(
                    {
                        "direction": direction,
                        "metric": body["metric"],
                        "bucket": bucket_name,
                        **ex,
                    }
                )
    return rows
