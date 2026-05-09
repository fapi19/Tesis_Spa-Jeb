"""Phase 5 runner: full evaluation of an NLLB+LoRA checkpoint on a split.

Generates beam-search top-1 + top-K predictions, then computes BLEU, chrF++,
BERTScore, and COMET per direction, with explicit Shiwilu-OOD caveats.

Usage:
    python scripts/nmt/40_evaluate.py --checkpoint models/nmt/nllb_bidi_lora_v0 --split test
    python scripts/nmt/40_evaluate.py --checkpoint models/nmt/nllb_bidi_lora_v0 --split test --skip-comet
    python scripts/nmt/40_evaluate.py --prewarm-only
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

import yaml  # noqa: E402

from src.nmt.evaluation.metrics import MetricsConfig, evaluate_predictions, prewarm_models  # noqa: E402
from src.nmt.inference.confidence import (  # noqa: E402
    attach_baseline_confidence,
    summarize_bands,
)
from src.nmt.inference.generate import GenerationConfig, load_checkpoint, predict_split  # noqa: E402

INFERENCE_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "inference.yaml"
EVAL_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "eval.yaml"
TRAINING_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "training.yaml"
FILTERED_DIR = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--prewarm-only", action="store_true")
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-comet", action="store_true")
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Override report directory (default reports/05_nmt/evaluation/<run>/).",
    )
    return p.parse_args()


_TOP1_DROPPED_KEYS = {"candidates"}


def _strip_predictions_for_jsonl(predictions: list[dict]) -> list[dict]:
    """Top-1 only (drop the candidates list to keep the file small)."""
    out = []
    for p in predictions:
        out.append({k: v for k, v in p.items() if k not in _TOP1_DROPPED_KEYS})
    return out


def main() -> int:
    args = parse_args()
    eval_cfg = MetricsConfig.from_yaml(EVAL_CFG_PATH)

    if args.prewarm_only:
        prewarm_models(eval_cfg)
        return 0

    if args.checkpoint is None:
        print("[phase5] --checkpoint is required (unless --prewarm-only)", file=sys.stderr)
        return 2

    checkpoint_path = Path(args.checkpoint).resolve()
    run_name = checkpoint_path.name
    report_dir = (
        Path(args.report) if args.report else PROJECT_ROOT / "reports" / "05_nmt" / "evaluation" / run_name
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase5] checkpoint={checkpoint_path}")
    print(f"[phase5] report_dir={report_dir.relative_to(PROJECT_ROOT)}")

    gen_cfg = GenerationConfig.from_yaml(INFERENCE_CFG_PATH)
    with TRAINING_CFG_PATH.open(encoding="utf-8") as f:
        training_yaml = yaml.safe_load(f)
    base_model = training_yaml["base_model"]
    lang_code_map = {str(k): str(v) for k, v in training_yaml["data"]["lang_code_map"].items()}

    model, tokenizer, device = load_checkpoint(
        checkpoint_path, base_model=base_model, device="auto"
    )
    print(f"[phase5] device={device}, beam={gen_cfg.num_beams}, n_best={gen_cfg.num_return_sequences}")

    csv_path = FILTERED_DIR / f"{args.split}.csv"
    print(f"[phase5] generating predictions on {csv_path.relative_to(PROJECT_ROOT)} ...")
    predictions = predict_split(
        model,
        tokenizer,
        csv_path,
        cfg=gen_cfg,
        lang_code_map=lang_code_map,
        device=device,
    )
    print(f"[phase5] generated {len(predictions)} predictions")

    predictions, conf_cfg = attach_baseline_confidence(predictions)
    band_distribution = summarize_bands(predictions)
    print(
        f"[phase5] confidence ({conf_cfg.name}, thresholds=({conf_cfg.low_to_med}, "
        f"{conf_cfg.med_to_high})): {band_distribution['overall']}"
    )

    pred_jsonl = report_dir / f"{args.split}_predictions.jsonl"
    topk_jsonl = report_dir / f"{args.split}_predictions_topk.jsonl"
    with pred_jsonl.open("w", encoding="utf-8") as f:
        for p in _strip_predictions_for_jsonl(predictions):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with topk_jsonl.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[phase5] wrote {pred_jsonl.relative_to(PROJECT_ROOT)}")
    print(f"[phase5] wrote {topk_jsonl.relative_to(PROJECT_ROOT)}")

    # Free GPU before loading metric models
    del model
    import torch
    torch.cuda.empty_cache()

    print("[phase5] computing metrics ...")
    metrics = evaluate_predictions(
        predictions,
        eval_cfg,
        include_bertscore=not args.skip_bertscore,
        include_comet=not args.skip_comet,
    )
    metrics["meta"] = {
        "phase": 5,
        "run_name": run_name,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "n_predictions": len(predictions),
        "generation": {
            "num_beams": gen_cfg.num_beams,
            "length_penalty": gen_cfg.length_penalty,
            "max_new_tokens": gen_cfg.max_new_tokens,
            "num_return_sequences": gen_cfg.num_return_sequences,
        },
        "confidence": {
            "thresholds": conf_cfg.as_dict(),
            "distribution": band_distribution,
        },
    }

    out_path = report_dir / f"{args.split}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase5] wrote {out_path.relative_to(PROJECT_ROOT)}")

    print("[phase5] headline metrics:")
    for direction, m in metrics["directions"].items():
        print(
            f"[phase5]   {direction}: chrF++={m.get('chrf_pp', float('nan')):.2f} "
            f"BLEU={m.get('bleu', float('nan')):.2f}"
        )
    if "avg_chrf_pp" in metrics:
        print(f"[phase5]   avg chrF++ = {metrics['avg_chrf_pp']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
