"""Phase 7c runner: generate morphological variants (off-by-default).

Outputs:
    reports/05_nmt/augmentation/morph_variants.json
    reports/05_nmt/augmentation/morph_variants_review.csv  (always written)
    data/processed/07_nmt_augmented/train_morph.csv         (only if
        --emit-csv-anyway is passed; default behavior is to NOT mix these
        non-validated rows into v1_bt training data).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.nmt.augmentation.morphological_variants import (  # noqa: E402
    MorphVariantConfig,
    generate_variants,
)

PARALLEL_CSV = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered" / "train.csv"
SUFFIXES_PATH = PROJECT_ROOT / "data" / "processed" / "04_splits" / "shiwilu_suffixes.json"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "processed" / "07_nmt_augmented"
REPORTS_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "augmentation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-n-suffixes", type=int, default=10)
    p.add_argument("--max-variants-per-word", type=int, default=3)
    p.add_argument(
        "--emit-csv-anyway",
        action="store_true",
        help="Write data/processed/07_nmt_augmented/train_morph.csv "
             "(NOT recommended without linguist supervision).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = MorphVariantConfig(
        top_n_suffixes=args.top_n_suffixes,
        max_variants_per_word=args.max_variants_per_word,
    )
    df, info = generate_variants(PARALLEL_CSV, SUFFIXES_PATH, cfg)
    info["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    review_csv = REPORTS_DIR / "morph_variants_review.csv"
    df.to_csv(review_csv, index=False, encoding="utf-8-sig")
    info["review_csv"] = str(review_csv.relative_to(PROJECT_ROOT))
    print(f"[phase7c] {info['variants_generated']} variants emitted to {review_csv.relative_to(PROJECT_ROOT)}")
    print(
        f"[phase7c] eligible single-word pairs: {info['eligible_single_word_pairs']}, "
        f"with detected suffix: {info['with_detected_suffix']}"
    )

    if args.emit_csv_anyway and not df.empty:
        AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for i, r in df.iterrows():
            pair_id = f"MORPH{i:06d}"
            for src_lang, tgt_lang, src, tgt in (
                ("shw", "spa", r["shiwilu_variant"], r["spanish_kept"]),
                ("spa", "shw", r["spanish_kept"], r["shiwilu_variant"]),
            ):
                rows.append(
                    {
                        "id": f"{pair_id}__{src_lang}2{tgt_lang}",
                        "pair_id": pair_id,
                        "group_id": f"GMORPH{i:06d}",
                        "source": src,
                        "target": tgt,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "split": "train",
                        "has_audit_flags": True,    # always flag morph rows
                        "origin_source": "morph_variant_unvalidated",
                        "score": float("nan"),
                        "label": "manual_review_required",
                    }
                )
        out = pd.DataFrame(rows)
        out_csv = AUGMENTED_DIR / "train_morph.csv"
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        info["wrote_train_morph_csv"] = str(out_csv.relative_to(PROJECT_ROOT))
        info["default_emit_csv"] = True
        print(f"[phase7c] WROTE {out_csv.relative_to(PROJECT_ROOT)} (--emit-csv-anyway, unvalidated)")
    else:
        info["wrote_train_morph_csv"] = None

    out_path = REPORTS_DIR / "morph_variants.json"
    out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase7c] report -> {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
