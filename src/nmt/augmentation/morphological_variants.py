"""Phase 7c: morphological-variant augmentation (off by default).

Produces candidate variants of single-word Shiwilu pairs by swapping the
detected suffix for a high-frequency alternative from the inventory at
data/processed/04_splits/shiwilu_suffixes.json.

Per plan section 37 the linguistically-validated mapping must come from a
linguist; without it, the generated variants are NOT trustworthy. By
default this module emits a "manual_review_required" status into the run
report and does NOT write a train_morph.csv output. Pass --emit-csv-anyway
on the command line if you accept the risk (e.g. for ablation purposes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class MorphVariantConfig:
    top_n_suffixes: int = 10
    max_variants_per_word: int = 3


def load_suffix_inventory(path: Path, top_n: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"unexpected suffix inventory format at {path}")
    return data[:top_n]


def detect_suffix(word: str, inventory: Sequence[dict]) -> str | None:
    """Return the longest suffix from the inventory that ends `word`, or None."""
    candidates = [str(item["suffix"]) for item in inventory]
    candidates.sort(key=len, reverse=True)
    for suf in candidates:
        if word.endswith(suf) and len(word) > len(suf):
            return suf
    return None


def is_single_word_pair(shiwilu: str, spanish: str) -> bool:
    return (
        len(shiwilu.split()) == 1
        and len(spanish.split()) <= 4
        and "." not in spanish[:-1]
    )


def generate_variants(
    parallel_csv: Path,
    suffixes_path: Path,
    cfg: MorphVariantConfig,
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(parallel_csv, encoding="utf-8-sig")
    pairs = df[df["source_lang"] == "shw"][["pair_id", "source", "target"]].drop_duplicates(subset=["pair_id"])
    pairs = pairs.rename(columns={"source": "shiwilu", "target": "spanish"}).reset_index(drop=True)

    inventory = load_suffix_inventory(suffixes_path, cfg.top_n_suffixes)
    inv_suffixes = [str(item["suffix"]) for item in inventory]

    rows: list[dict] = []
    n_eligible = 0
    n_with_suffix = 0
    for _, row in pairs.iterrows():
        shw_word = str(row["shiwilu"]).strip()
        spa = str(row["spanish"]).strip()
        if not is_single_word_pair(shw_word, spa):
            continue
        n_eligible += 1
        suf = detect_suffix(shw_word, inventory)
        if suf is None:
            continue
        n_with_suffix += 1
        stem = shw_word[: -len(suf)]
        # Generate candidate replacements: swap with the next high-frequency suffixes.
        replacements = [s for s in inv_suffixes if s != suf][: cfg.max_variants_per_word]
        for r in replacements:
            variant = stem + r
            rows.append(
                {
                    "source_pair_id": row["pair_id"],
                    "shiwilu_original": shw_word,
                    "shiwilu_variant": variant,
                    "detected_suffix": suf,
                    "swapped_with": r,
                    "spanish_kept": spa,
                    "status": "manual_review_required",
                    "warning": (
                        "morphological mapping not linguist-validated; do NOT add to "
                        "training data without supervision"
                    ),
                }
            )
    info = {
        "phase": "7c",
        "step": "morph_variants",
        "eligible_single_word_pairs": n_eligible,
        "with_detected_suffix": n_with_suffix,
        "variants_generated": int(len(rows)),
        "default_emit_csv": False,
        "rationale": (
            "Plan section 37: morphological variants require linguist supervision "
            "to validate the suffix-swap mapping. Without it, the generated rows "
            "are pre-flagged manual_review_required and excluded from the v1_bt "
            "training set by default."
        ),
        "config": {
            "top_n_suffixes": cfg.top_n_suffixes,
            "max_variants_per_word": cfg.max_variants_per_word,
        },
    }
    return pd.DataFrame(rows), info
