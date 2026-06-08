"""Phase 5c runner: stratified qualitative analysis of a single NMT run.

Following the MT-evaluation expert's advice, this buckets the reranked (or
baseline) predictions of one run by a *per-sentence* score and writes
representative examples per bucket. shw2spa is scored by sentence-BLEU,
spa2shw by sentence-chrF++ (see :mod:`src.nmt.evaluation.qualitative`).

No GPU / model re-run is needed: it consumes predictions already on disk.

Usage:
    # Champion, reranked (default)
    python -m scripts.nmt.42_qualitative_analysis --variant xl
    # A specific run, baseline predictions
    python -m scripts.nmt.42_qualitative_analysis --variant xl \
        --run nllb_bidi_lora_v0 --no-reranked
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nmt._paths import resolve_paths  # noqa: E402
from src.nmt.evaluation.metrics import MetricsConfig  # noqa: E402
from src.nmt.evaluation.qualitative import (  # noqa: E402
    QualitativeConfig,
    evaluate_qualitative,
    sampled_rows,
)
from src.nmt.evaluation.rare_token import attach_origin_source  # noqa: E402

EVAL_CFG_PATH = PROJECT_ROOT / "config" / "nmt" / "eval.yaml"
DEFAULT_RUN = "nllb_bidi_lora_v2_1b_loraplus_xl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="xl")
    p.add_argument("--run", type=str, default=DEFAULT_RUN, help="Run name (champion by default).")
    p.add_argument(
        "--reranked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reranked predictions (default). --no-reranked uses baseline.",
    )
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--samples-per-bucket", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--predictions", type=str, default=None, help="Override predictions JSONL.")
    p.add_argument("--test-csv", type=str, default=None, help="Override test CSV (for origin_source).")
    p.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    return p.parse_args()


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    nmt = resolve_paths(PROJECT_ROOT, args.variant)
    suffix = "_xl" if args.variant == "xl" else ""
    if args.predictions is not None:
        predictions_path = Path(args.predictions).resolve()
    else:
        sub = f"reranking{suffix}" if args.reranked else f"evaluation{suffix}"
        filename = (
            f"{args.split}_predictions_reranked.jsonl"
            if args.reranked
            else f"{args.split}_predictions.jsonl"
        )
        predictions_path = PROJECT_ROOT / "reports" / "05_nmt" / sub / args.run / filename

    test_path = Path(args.test_csv).resolve() if args.test_csv else nmt.filtered_dir / "test.csv"

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = nmt.reports_evaluation_dir / args.run / "qualitative"
    return predictions_path, test_path, out_dir


def _fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}\\%".replace("\\", "")


def build_markdown(report: dict, run: str, reranked: bool) -> str:
    lines: list[str] = []
    lines.append(f"# Análisis cualitativo estratificado — {run}")
    lines.append("")
    lines.append(f"- Predicciones: {'reranked' if reranked else 'baseline'}")
    lines.append(f"- Muestras por bucket: {report['config']['samples_per_bucket']} (seed {report['config']['seed']})")
    lines.append("")
    lines.append(
        "shw→spa se puntúa con BLEU por oración; spa→shw con chrF++ por oración. "
        "Cortes según el umbral de ruido del experto (BLEU≤10 / chrF++≤20 ≈ ruido)."
    )
    lines.append("")
    for direction, body in report["directions"].items():
        lines.append(f"## {direction} (métrica: {body['metric']}, n={body['n']}, media={body['mean_score']:.2f})")
        lines.append("")
        lines.append("| Bucket | n | % | media |")
        lines.append("|---|---:|---:|---:|")
        for name, st in body["buckets"].items():
            lines.append(f"| {name} | {st['n']} | {_fmt_pct(st['pct'])} | {st['mean_score']:.2f} |")
        lines.append("")
        for name, examples in body["samples"].items():
            if not examples:
                continue
            lines.append(f"### Ejemplos — {name}")
            for ex in examples:
                lines.append(f"- **score={ex['sentence_score']}** (`{ex['origin_source']}`)")
                lines.append(f"  - fuente: {ex['source']}")
                lines.append(f"  - referencia: {ex['reference']}")
                lines.append(f"  - hipótesis: {ex['hypothesis']}")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    predictions_path, test_path, out_dir = _resolve_paths(args)

    if not predictions_path.exists():
        print(f"[phase5c] missing predictions: {predictions_path}", file=sys.stderr)
        return 2

    print(f"[phase5c] predictions={predictions_path.relative_to(PROJECT_ROOT)}")
    predictions = load_predictions(predictions_path)
    print(f"[phase5c] loaded {len(predictions)} predictions")
    predictions = attach_origin_source(predictions, test_path)

    metrics_cfg = MetricsConfig.from_yaml(EVAL_CFG_PATH)
    qual_cfg = QualitativeConfig(
        samples_per_bucket=int(args.samples_per_bucket),
        seed=int(args.seed),
    )
    report = evaluate_qualitative(predictions, metrics_cfg, qual_cfg)
    report["meta"] = {
        "phase": "5c",
        "run_name": args.run,
        "split": args.split,
        "reranked": bool(args.reranked),
        "predictions_path": str(predictions_path),
        "test_path": str(test_path),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "bucket_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = sampled_rows(report)
    csv_path = out_dir / "sampled_examples.csv"
    fieldnames = ["direction", "metric", "bucket", "sentence_score", "origin_source",
                  "id", "pair_id", "source", "reference", "hypothesis"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_path = out_dir / "qualitative_report.md"
    md_path.write_text(build_markdown(report, args.run, bool(args.reranked)), encoding="utf-8")

    print(f"[phase5c] wrote {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase5c] wrote {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase5c] wrote {md_path.relative_to(PROJECT_ROOT)}")
    for direction, body in report["directions"].items():
        dist = ", ".join(f"{n}={st['n']}" for n, st in body["buckets"].items())
        print(f"[phase5c]   {direction} ({body['metric']}): {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
