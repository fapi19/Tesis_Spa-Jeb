"""Phase 7a runner: build monolingual Shiwilu pool, backtranslate via v0,
filter through the Phase 2 semantic filter, write data/processed/07_nmt_augmented/train_bt.csv.

Usage:
    # Step 1: extract monolingual Shiwilu pool (no model needed).
    python scripts/nmt/60_backtranslate.py --extract-mono \
        --candidate data/raw/II_TEXTOS_SHIWILU_extracted.txt

    # Step 2: backtranslate using a frozen v0 checkpoint.
    python scripts/nmt/60_backtranslate.py --checkpoint models/nmt/nllb_bidi_lora_v0
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

from src.nmt.augmentation.backtranslation import (  # noqa: E402
    BackTranslationConfig,
    cap_synthetic,
    extract_mono_shiwilu,
    make_synthetic_dataframe,
)

PARALLEL_DIR = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "processed" / "07_nmt_augmented"
REPORTS_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "augmentation"
DEFAULT_MONO = AUGMENTED_DIR / "mono_shw.txt"
DEFAULT_TRAIN_BT = AUGMENTED_DIR / "train_bt.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract-mono", action="store_true", help="Stage 1: extract mono pool only.")
    p.add_argument(
        "--candidate",
        action="append",
        type=str,
        default=[],
        help="Candidate text file with line-per-sentence Shiwilu sources. Repeatable.",
    )
    p.add_argument("--mono", type=str, default=str(DEFAULT_MONO))
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--no-require-apostrophe",
        action="store_true",
        help="Relax the Shiwilu heuristic to allow lines without apostrophes (more recall, more Spanish FPs).",
    )
    p.add_argument("--accept-threshold", type=float, default=0.60)
    p.add_argument("--cap-x", type=float, default=2.0, help="Cap synthetic at this multiple of parallel size.")
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Override report directory (default reports/05_nmt/augmentation/).",
    )
    return p.parse_args()


def _stage_extract_mono(args: argparse.Namespace) -> int:
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_paths = [Path(c).resolve() for c in args.candidate]
    parallel_csvs = [
        PARALLEL_DIR / "train.csv",
        PARALLEL_DIR / "valid.csv",
        PARALLEL_DIR / "test.csv",
    ]
    print(f"[phase7a] candidates: {[str(p.relative_to(PROJECT_ROOT)) for p in candidate_paths] or '<none>'}")
    info = extract_mono_shiwilu(
        parallel_csvs,
        candidate_paths,
        Path(args.mono),
        require_apostrophe=not args.no_require_apostrophe,
    )
    info["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    info["mode"] = "extract_mono"
    print(
        f"[phase7a] kept {info['kept']} mono Shiwilu lines "
        f"(skipped {info['skipped_in_parallel']} in_parallel, "
        f"{info['skipped_looks_spanish']} looks_spanish)"
    )

    out_dir = Path(args.report) if args.report else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mono_extraction.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


def _stage_backtranslate(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        print("[phase7a] --checkpoint is required for backtranslation stage", file=sys.stderr)
        return 2

    mono_path = Path(args.mono)
    if not mono_path.exists() or mono_path.stat().st_size == 0:
        print(
            f"[phase7a] mono pool empty or missing: {mono_path}\n"
            f"          run --extract-mono with --candidate first.",
            file=sys.stderr,
        )
        return 3

    with mono_path.open(encoding="utf-8") as f:
        mono_lines = [ln.strip() for ln in f if ln.strip()]
    print(f"[phase7a] mono lines: {len(mono_lines)}")

    # Lazy-imported here so the extract-mono stage doesn't pay the model load cost.
    import torch  # noqa: F401
    import yaml
    from src.nmt.evaluation.metrics import MetricsConfig  # noqa: F401  (config peek)
    from src.nmt.inference.generate import GenerationConfig, load_checkpoint, generate_for_direction
    from src.nmt.preprocessing.semantic_filter import (
        SemanticFilterConfig,
        compute_pair_scores,
        load_embedding_model,
    )

    with (PROJECT_ROOT / "config" / "nmt" / "training.yaml").open(encoding="utf-8") as f:
        training_yaml = yaml.safe_load(f)
    base_model = training_yaml["base_model"]
    lang_code_map = {str(k): str(v) for k, v in training_yaml["data"]["lang_code_map"].items()}

    gen_cfg = GenerationConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "inference.yaml")
    print(f"[phase7a] loading checkpoint {args.checkpoint}")
    model, tokenizer, device = load_checkpoint(Path(args.checkpoint), base_model=base_model, device="auto")

    df_mono = pd.DataFrame(
        {
            "id": [f"BTRAW{i:06d}__shw2spa" for i in range(len(mono_lines))],
            "pair_id": [f"BTRAW{i:06d}" for i in range(len(mono_lines))],
            "source": mono_lines,
            "target": [""] * len(mono_lines),
        }
    )
    print(f"[phase7a] generating Spanish for {len(df_mono)} mono lines (beam={gen_cfg.num_beams}) ...")
    predictions = generate_for_direction(
        model,
        tokenizer,
        df_mono,
        src_plan="shw",
        tgt_plan="spa",
        lang_code_map=lang_code_map,
        cfg=gen_cfg,
        device=device,
        return_topk=False,
    )
    spanish = [p["hypothesis"] for p in predictions]

    # Free LoRA / NLLB before loading SBERT.
    del model
    import torch as _torch
    _torch.cuda.empty_cache()

    print("[phase7a] scoring synthetic pairs with v3 SBERT ...")
    filter_cfg = SemanticFilterConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "filter.yaml", PROJECT_ROOT)
    sbert = load_embedding_model(filter_cfg)
    scores = compute_pair_scores(
        sbert,
        mono_lines,
        spanish,
        batch_size=filter_cfg.batch_size,
        use_e5=filter_cfg.use_e5_prefixes,
    )

    syn_df = make_synthetic_dataframe(
        mono_lines, spanish, [float(s) for s in scores], accept_threshold=args.accept_threshold
    )
    print(f"[phase7a] synthetic rows after threshold {args.accept_threshold}: {len(syn_df)}")

    parallel_train = pd.read_csv(PARALLEL_DIR / "train.csv", encoding="utf-8-sig")
    capped = cap_synthetic(syn_df, parallel_size=len(parallel_train), cfg=BackTranslationConfig(bt_cap_x_parallel=args.cap_x))
    print(f"[phase7a] capped synthetic rows: {len(capped)} (cap={args.cap_x}x parallel of size {len(parallel_train)})")

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    capped.to_csv(DEFAULT_TRAIN_BT, index=False, encoding="utf-8-sig")
    print(f"[phase7a] wrote {DEFAULT_TRAIN_BT.relative_to(PROJECT_ROOT)}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "7a",
        "step": "backtranslation",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mono_lines_in": len(mono_lines),
        "synthetic_rows_after_filter": int(len(syn_df)),
        "synthetic_rows_after_cap": int(len(capped)),
        "accept_threshold": args.accept_threshold,
        "cap_x_parallel": args.cap_x,
        "parallel_train_size": int(len(parallel_train)),
        "score_stats": {
            "mean": float(syn_df["score"].mean()) if len(syn_df) else None,
            "min": float(syn_df["score"].min()) if len(syn_df) else None,
            "max": float(syn_df["score"].max()) if len(syn_df) else None,
        },
    }
    out_dir = Path(args.report) if args.report else REPORTS_DIR
    (out_dir / "backtranslation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[phase7a] report -> {(out_dir / 'backtranslation.json').relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    args = parse_args()
    if args.extract_mono:
        return _stage_extract_mono(args)
    return _stage_backtranslate(args)


if __name__ == "__main__":
    raise SystemExit(main())
