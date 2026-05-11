"""Phase 6 runner: semantic reranking on the top-K predictions of a checkpoint.

Reads reports/05_nmt/evaluation/<run>/<split>_predictions_topk.jsonl, applies
the configured 0.7 / 0.3 weighting (translation prob / cosine), and runs the
alpha ablation sweep.

Outputs into reports/05_nmt/reranking/<run>/:
    <split>_predictions_reranked.jsonl
    <split>_metrics_reranked.json
    ablation.json
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

from src.nmt.evaluation.metrics import MetricsConfig, evaluate_predictions  # noqa: E402
from src.nmt.inference.confidence import (  # noqa: E402
    attach_reranked_confidence,
    summarize_bands,
)
from src.nmt.reranking.semantic_reranker import (  # noqa: E402
    RerankerConfig,
    evaluate_rerank_ablation,
    rerank,
)
from scripts.nmt._paths import resolve_paths

RERANK_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "reranker.yaml"
EVAL_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "eval.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="main")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Single checkpoint path. Mutually exclusive with --run-name (Two-DoRA case).")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run name (e.g. 'v3_two_dora_xl'). Use when reranking a Two-DoRA combined eval that has no single checkpoint path.")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-comet", action="store_true")
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Override report directory (default reports/05_nmt/reranking/<run>/).",
    )
    p.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Override topk predictions path (default reports/05_nmt/evaluation/<run>/<split>_predictions_topk.jsonl).",
    )
    return p.parse_args()


def load_topk(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    args = parse_args()
    nmt_paths = resolve_paths(PROJECT_ROOT, args.variant)

    if args.checkpoint is None and args.run_name is None:
        print("[phase6] either --checkpoint or --run-name is required", file=sys.stderr)
        return 2

    if args.run_name is not None:
        run_name = args.run_name
        checkpoint_path = None
        checkpoint_repr = f"<two_dora_run:{run_name}>"
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        run_name = checkpoint_path.name
        checkpoint_repr = str(checkpoint_path)

    if args.predictions is None:
        predictions_path = nmt_paths.reports_evaluation_dir / run_name / f"{args.split}_predictions_topk.jsonl"
    else:
        predictions_path = Path(args.predictions).resolve()

    if not predictions_path.exists():
        print(f"[phase6] missing top-k predictions: {predictions_path}", file=sys.stderr)
        return 2

    report_dir = Path(args.report) if args.report else nmt_paths.reports_reranking_dir / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase6] loading {predictions_path.relative_to(PROJECT_ROOT)}")
    predictions = load_topk(predictions_path)
    print(f"[phase6] {len(predictions)} predictions, candidates per row = "
          f"{len(predictions[0]['candidates']) if predictions else 0}")

    rcfg = RerankerConfig.from_yaml(RERANK_CFG_PATH, PROJECT_ROOT)
    if args.variant == "xl":
        object.__setattr__(
            rcfg,
            "embedding_path",
            PROJECT_ROOT / "models" / "sentence_transformers" / "v3_iterative_hn_e5_base_bidirectional_xl",
        )
    eval_cfg = MetricsConfig.from_yaml(EVAL_CFG_PATH)
    print(
        f"[phase6] variant={args.variant}, reranker weights: trans={rcfg.weight_translation}, "
        f"sem={rcfg.weight_semantic}, alpha sweep={list(rcfg.ablation_alphas)}"
    )

    reranked = rerank(predictions, rcfg)
    reranked, conf_cfg = attach_reranked_confidence(reranked, alpha=rcfg.weight_translation)
    band_distribution = summarize_bands(reranked)
    print(
        f"[phase6] confidence ({conf_cfg.name}, thresholds=({conf_cfg.low_to_med}, "
        f"{conf_cfg.med_to_high})): {band_distribution['overall']}"
    )

    out_jsonl = report_dir / f"{args.split}_predictions_reranked.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in reranked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[phase6] wrote {out_jsonl.relative_to(PROJECT_ROOT)}")

    metrics = evaluate_predictions(
        reranked,
        eval_cfg,
        include_bertscore=not args.skip_bertscore,
        include_comet=not args.skip_comet,
    )
    metrics["meta"] = {
        "phase": 6,
        "run_name": run_name,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": checkpoint_repr,
        "split": args.split,
        "n_predictions": len(reranked),
        "alpha": rcfg.weight_translation,
        "embedding_model": str(rcfg.embedding_path.relative_to(PROJECT_ROOT)),
        "confidence": {
            "thresholds": conf_cfg.as_dict(),
            "distribution": band_distribution,
        },
    }
    metrics_path = report_dir / f"{args.split}_metrics_reranked.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase6] wrote {metrics_path.relative_to(PROJECT_ROOT)}")

    print("[phase6] running alpha ablation ...")
    ablation = evaluate_rerank_ablation(predictions, rcfg, eval_cfg)
    ablation["meta"] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "split": args.split,
        "checkpoint": checkpoint_repr,
    }
    ablation_path = report_dir / "ablation.json"
    ablation_path.write_text(json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase6] wrote {ablation_path.relative_to(PROJECT_ROOT)}")

    print("[phase6] reranked headline metrics:")
    for direction, m in metrics["directions"].items():
        print(
            f"[phase6]   {direction}: chrF++={m.get('chrf_pp', float('nan')):.2f} "
            f"BLEU={m.get('bleu', float('nan')):.2f}"
        )
    if "avg_chrf_pp" in metrics:
        print(f"[phase6]   avg chrF++ = {metrics['avg_chrf_pp']:.2f}")
    print(f"[phase6] best alpha (by avg chrF++): {ablation['best']['alpha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
