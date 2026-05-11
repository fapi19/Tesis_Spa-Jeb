"""Phase 2a runner: apply the v3 SBERT semantic filter to canonical CSVs.

Output:
    data/processed/06_nmt_filtered/{train,train_flagged,train_removed,valid,test}.csv
    reports/05_nmt/preprocessing/semantic_filter.json

Train rows are split into accepted/flagged/removed by per-pair_id score.
Valid and test pass through with the score column attached (gold splits stay
frozen, only audited).
"""
from __future__ import annotations

import datetime as dt
import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.nmt.preprocessing.semantic_filter import (  # noqa: E402
    SemanticFilterConfig,
    histogram,
    load_embedding_model,
    per_origin_stats,
    resolve_device,
    score_split,
    write_partition,
)
from scripts.nmt._paths import resolve_paths

CONFIG_PATH = PROJECT_ROOT / "config" / "nmt" / "filter.yaml"
HIST_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
TOPK_WORST = 25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["main", "xl"], default="main")
    args = parser.parse_args()
    nmt_paths = resolve_paths(PROJECT_ROOT, args.variant)
    canon_dir = nmt_paths.canonical_dir
    out_dir = nmt_paths.filtered_dir
    reports_dir = nmt_paths.reports_preprocessing_dir

    cfg = SemanticFilterConfig.from_yaml(CONFIG_PATH, PROJECT_ROOT)
    if args.variant == "xl":
        object.__setattr__(
            cfg,
            "model_path",
            PROJECT_ROOT / "models" / "sentence_transformers" / "v3_iterative_hn_e5_base_bidirectional_xl",
        )
    device = resolve_device(cfg.device)
    print(f"[phase2a] variant={args.variant}, device={device}, model={cfg.model_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase2a] thresholds: remove<{cfg.thresholds.remove_below} flag<={cfg.thresholds.flag_upper}")

    model = load_embedding_model(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "phase": "2a",
        "step": "semantic_filter",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "model_path": str(cfg.model_path.relative_to(PROJECT_ROOT)),
            "use_e5_prefixes": cfg.use_e5_prefixes,
            "batch_size": cfg.batch_size,
            "device": device,
            "thresholds": {
                "remove_below": cfg.thresholds.remove_below,
                "flag_upper": cfg.thresholds.flag_upper,
            },
        },
        "splits": {},
    }

    # ---- TRAIN: filter into accepted/flagged/removed ---------------------
    train_csv = canon_dir / "train.csv"
    print(f"[phase2a] scoring TRAIN ({train_csv.relative_to(PROJECT_ROOT)})")
    train_df, train_pairs = score_split(train_csv, model, cfg)

    accepted = train_df[train_df["label"] == "accepted"].copy()
    flagged = train_df[train_df["label"] == "flagged_for_review"].copy()
    removed = train_df[train_df["label"] == "removed"].copy()

    write_partition(accepted, out_dir, "train")
    write_partition(flagged, out_dir, "train_flagged")
    write_partition(removed, out_dir, "train_removed")

    print(
        f"[phase2a]   train rows: accepted={len(accepted)} "
        f"flagged={len(flagged)} removed={len(removed)} (pairs={len(train_pairs)})"
    )

    train_scores = train_pairs["score"].to_numpy()
    report["splits"]["train"] = {
        "input_pairs": int(len(train_pairs)),
        "directional_rows_in": int(len(train_df)),
        "directional_rows_accepted": int(len(accepted)),
        "directional_rows_flagged": int(len(flagged)),
        "directional_rows_removed": int(len(removed)),
        "label_counts_pairs": {
            label: int((train_pairs["label"] == label).sum())
            for label in ("accepted", "flagged_for_review", "removed")
        },
        "score_stats": {
            "mean": float(train_scores.mean()),
            "std": float(train_scores.std(ddof=0)),
            "min": float(train_scores.min()),
            "max": float(train_scores.max()),
            "p25": float(pd.Series(train_scores).quantile(0.25)),
            "p50": float(pd.Series(train_scores).quantile(0.50)),
            "p75": float(pd.Series(train_scores).quantile(0.75)),
        },
        "histogram": histogram(train_scores, HIST_BINS),
        "per_origin": per_origin_stats(train_pairs, train_df),
        "topk_worst": train_pairs.sort_values("score").head(TOPK_WORST)[
            ["pair_id", "shiwilu", "spanish", "score", "label"]
        ].to_dict(orient="records"),
    }

    # ---- VALID + TEST: passthrough with score column ---------------------
    for split in ("valid", "test"):
        path = canon_dir / f"{split}.csv"
        print(f"[phase2a] scoring {split.upper()} ({path.relative_to(PROJECT_ROOT)})")
        df, pairs = score_split(path, model, cfg)
        write_partition(df, out_dir, split)
        scores = pairs["score"].to_numpy()
        report["splits"][split] = {
            "input_pairs": int(len(pairs)),
            "directional_rows": int(len(df)),
            "score_stats": {
                "mean": float(scores.mean()),
                "std": float(scores.std(ddof=0)),
                "min": float(scores.min()),
                "max": float(scores.max()),
            },
            "label_counts_pairs": {
                label: int((pairs["label"] == label).sum())
                for label in ("accepted", "flagged_for_review", "removed")
            },
            "histogram": histogram(scores, HIST_BINS),
            "note": "passthrough; gold splits never filtered, only scored for audit",
        }
        print(
            f"[phase2a]   {split} mean={scores.mean():.4f} std={scores.std(ddof=0):.4f} "
            f"flagged_pairs={int((pairs['label'] == 'flagged_for_review').sum())} "
            f"removed_pairs={int((pairs['label'] == 'removed').sum())}"
        )

    out_path = reports_dir / "semantic_filter.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase2a] report -> {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
