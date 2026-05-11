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
from scripts.nmt._paths import resolve_paths

INFERENCE_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "inference.yaml"
EVAL_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "eval.yaml"
TRAINING_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "training.yaml"
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="main")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Single bidirectional checkpoint. Mutually exclusive with --checkpoint-spa2shw/--checkpoint-shw2spa.")
    p.add_argument("--checkpoint-spa2shw", type=str, default=None,
                   help="Two-DoRA mode: checkpoint for spa->shw direction.")
    p.add_argument("--checkpoint-shw2spa", type=str, default=None,
                   help="Two-DoRA mode: checkpoint for shw->spa direction.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Required in Two-DoRA mode (combined report dir name).")
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


def _is_two_dora(args: argparse.Namespace) -> bool:
    return args.checkpoint_spa2shw is not None or args.checkpoint_shw2spa is not None


def _generate_two_dora(
    args: argparse.Namespace,
    *,
    base_model: str,
    lang_code_map: dict[str, str],
    csv_path: Path,
    gen_cfg,
):
    """Two-DoRA inference: load each adapter in turn and generate its direction."""
    from src.nmt.inference.generate import generate_for_direction
    import pandas as pd
    import torch

    if args.checkpoint_spa2shw is None or args.checkpoint_shw2spa is None:
        raise SystemExit("[phase5] Two-DoRA mode requires BOTH --checkpoint-spa2shw and --checkpoint-shw2spa")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(subset=["source", "target"]).reset_index(drop=True)
    out = []

    for direction, ckpt_path in (("spa2shw", args.checkpoint_spa2shw), ("shw2spa", args.checkpoint_shw2spa)):
        src_plan, tgt_plan = direction.split("2")
        sub = df[(df["source_lang"] == src_plan) & (df["target_lang"] == tgt_plan)].copy()
        if len(sub) == 0:
            continue
        ckpt = Path(ckpt_path).resolve()
        print(f"[phase5] [two-dora] loading {direction} adapter from {ckpt}")
        model, tokenizer, device = load_checkpoint(ckpt, base_model=base_model, device="auto")
        out.extend(
            generate_for_direction(
                model, tokenizer, sub,
                src_plan=src_plan, tgt_plan=tgt_plan,
                lang_code_map=lang_code_map, cfg=gen_cfg, device=device,
                return_topk=True,
            )
        )
        del model
        torch.cuda.empty_cache()
    return out


_TOP1_DROPPED_KEYS = {"candidates"}


def _strip_predictions_for_jsonl(predictions: list[dict]) -> list[dict]:
    """Top-1 only (drop the candidates list to keep the file small)."""
    out = []
    for p in predictions:
        out.append({k: v for k, v in p.items() if k not in _TOP1_DROPPED_KEYS})
    return out


def main() -> int:
    args = parse_args()
    nmt_paths = resolve_paths(PROJECT_ROOT, args.variant)
    eval_cfg = MetricsConfig.from_yaml(EVAL_CFG_PATH)

    if args.prewarm_only:
        prewarm_models(eval_cfg)
        return 0

    two_dora = _is_two_dora(args)

    if not two_dora and args.checkpoint is None:
        print("[phase5] --checkpoint is required (unless --prewarm-only or Two-DoRA mode)", file=sys.stderr)
        return 2

    if two_dora:
        if args.run_name is None:
            print("[phase5] --run-name is required in Two-DoRA mode", file=sys.stderr)
            return 2
        run_name = args.run_name
        checkpoint_repr = f"two_dora(spa2shw={args.checkpoint_spa2shw}, shw2spa={args.checkpoint_shw2spa})"
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        run_name = checkpoint_path.name
        checkpoint_repr = str(checkpoint_path)

    report_dir = Path(args.report) if args.report else nmt_paths.reports_evaluation_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase5] variant={args.variant}, checkpoint={checkpoint_repr}")
    print(f"[phase5] report_dir={report_dir.relative_to(PROJECT_ROOT)}")

    gen_cfg = GenerationConfig.from_yaml(INFERENCE_CFG_PATH)
    with TRAINING_CFG_PATH.open(encoding="utf-8") as f:
        training_yaml = yaml.safe_load(f)
    base_model = training_yaml["base_model"]
    lang_code_map = {str(k): str(v) for k, v in training_yaml["data"]["lang_code_map"].items()}

    csv_path = nmt_paths.filtered_dir / f"{args.split}.csv"
    print(f"[phase5] generating predictions on {csv_path.relative_to(PROJECT_ROOT)} ...")

    if two_dora:
        predictions = _generate_two_dora(
            args, base_model=base_model, lang_code_map=lang_code_map,
            csv_path=csv_path, gen_cfg=gen_cfg,
        )
    else:
        model, tokenizer, device = load_checkpoint(
            checkpoint_path, base_model=base_model, device="auto"
        )
        print(f"[phase5] device={device}, beam={gen_cfg.num_beams}, n_best={gen_cfg.num_return_sequences}")
        predictions = predict_split(
            model, tokenizer, csv_path,
            cfg=gen_cfg, lang_code_map=lang_code_map, device=device,
        )
        del model
        import torch as _torch
        _torch.cuda.empty_cache()
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

    # GPU was already freed above (single or two-dora branch).

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
        "checkpoint": checkpoint_repr,
        "two_dora": two_dora,
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
