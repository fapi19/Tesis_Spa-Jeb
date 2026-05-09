"""Phase 5b runner: rare-token / morphology-aware evaluation (Enhancement #2).

Consumes the predictions JSONL emitted by Phase 5 (or Phase 6) and the
canonical training table, then bins each test row by the fraction of rare
words in its reference. chrF++ is recomputed per bucket; an
``oov_recovery_rate`` is reported alongside.

Usage:
    # Baseline v0
    python scripts/nmt/41_rare_token_eval.py --run nllb_bidi_lora_v0
    # Reranked v0
    python scripts/nmt/41_rare_token_eval.py --run nllb_bidi_lora_v0 --reranked
    # Override paths explicitly
    python scripts/nmt/41_rare_token_eval.py \
        --predictions reports/05_nmt/evaluation/<run>/test_predictions.jsonl \
        --train data/processed/06_nmt_filtered/train.csv \
        --output reports/05_nmt/evaluation/<run>/rare_token_analysis.json
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

from src.nmt.evaluation.metrics import MetricsConfig  # noqa: E402
from src.nmt.evaluation.rare_token import (  # noqa: E402
    RareTokenConfig,
    attach_origin_source,
    compute_train_freqs,
    evaluate_rare_tokens,
)

EVAL_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "eval.yaml"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered" / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered" / "test.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run name (e.g. nllb_bidi_lora_v0); resolves report paths automatically.",
    )
    p.add_argument(
        "--reranked",
        action="store_true",
        help="Use reranked predictions instead of baseline (requires Phase 6 output).",
    )
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--predictions", type=str, default=None, help="Override predictions JSONL.")
    p.add_argument("--train", type=str, default=None, help="Override training CSV.")
    p.add_argument("--test-csv", type=str, default=None, help="Override test CSV (for origin_source).")
    p.add_argument("--output", type=str, default=None, help="Override output JSON.")
    p.add_argument(
        "--rare-threshold",
        type=int,
        default=5,
        help="A word with train frequency strictly below this is treated as rare (default 5).",
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.predictions is not None:
        predictions_path = Path(args.predictions).resolve()
    else:
        if args.run is None:
            raise SystemExit("[phase5b] either --run or --predictions is required")
        sub = "reranking" if args.reranked else "evaluation"
        filename = (
            f"{args.split}_predictions_reranked.jsonl"
            if args.reranked
            else f"{args.split}_predictions.jsonl"
        )
        predictions_path = (
            PROJECT_ROOT / "reports" / "05_nmt" / sub / args.run / filename
        )

    train_path = Path(args.train).resolve() if args.train else DEFAULT_TRAIN
    test_path = Path(args.test_csv).resolve() if args.test_csv else DEFAULT_TEST

    if args.output is not None:
        out_path = Path(args.output).resolve()
    else:
        if args.run is None:
            raise SystemExit("[phase5b] --output is required when --run is omitted")
        out_dir = PROJECT_ROOT / "reports" / "05_nmt" / "evaluation" / args.run
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_reranked" if args.reranked else ""
        out_path = out_dir / f"rare_token_analysis{suffix}.json"

    return predictions_path, train_path, test_path, out_path


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    predictions_path, train_path, test_path, out_path = _resolve_paths(args)

    if not predictions_path.exists():
        print(f"[phase5b] missing predictions: {predictions_path}", file=sys.stderr)
        return 2
    if not train_path.exists():
        print(f"[phase5b] missing train CSV: {train_path}", file=sys.stderr)
        return 2

    print(f"[phase5b] predictions={predictions_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase5b] train={train_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase5b] test={test_path.relative_to(PROJECT_ROOT)}")

    predictions = load_predictions(predictions_path)
    print(f"[phase5b] loaded {len(predictions)} predictions")
    predictions = attach_origin_source(predictions, test_path)

    train_freqs = compute_train_freqs(train_path)
    for lang, c in train_freqs.items():
        print(f"[phase5b]   train vocab[{lang}]: {len(c)} unique words, {sum(c.values())} tokens")

    metrics_cfg = MetricsConfig.from_yaml(EVAL_CFG_PATH)
    rare_cfg = RareTokenConfig(rare_threshold=int(args.rare_threshold))

    report = evaluate_rare_tokens(predictions, train_freqs, metrics_cfg, rare_cfg)
    report["meta"] = {
        "phase": "5b",
        "run_name": args.run,
        "split": args.split,
        "reranked": args.reranked,
        "predictions_path": str(predictions_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase5b] wrote {out_path.relative_to(PROJECT_ROOT)}")

    print("[phase5b] headline rare-token chrF++ (>=20% rare bucket):")
    for direction, body in report["directions"].items():
        bucket = body["buckets"]["20pct_or_more"]
        oov = body["oov"]
        print(
            f"[phase5b]   {direction}: chrF++={bucket['chrf_pp']:.2f} (n={bucket['n']}), "
            f"oov_recovery={oov['recovery_rate']:.3f} ({oov['recovered']}/{oov['total']})"
        )
    if "avg_rare_chrf_pp_20pct_or_more" in report:
        print(f"[phase5b]   avg rare-bucket chrF++ = {report['avg_rare_chrf_pp_20pct_or_more']:.2f}")
        print(f"[phase5b]   avg oov_recovery       = {report['avg_oov_recovery_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
