"""Phase 8a runner: build the side-by-side comparison report for v0 / v1_bt
with and without semantic reranking.

Reads metric JSONs from:
    reports/05_nmt/evaluation/<run>/test_metrics.json
    reports/05_nmt/reranking/<run>/test_metrics_reranked.json

Emits:
    reports/05_nmt/evaluation/comparison_v0_vs_v1_bt.md

Robust to missing runs (e.g. before v1_bt has been trained): rows for
unavailable variants show '-'.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v0", default="nllb_bidi_lora_v0")
    p.add_argument("--v1", default="nllb_bidi_lora_v1_bt")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    return p.parse_args()


def _read_metric_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, metrics: dict | None, direction: str) -> str:
    def f(name: str) -> str:
        if metrics is None:
            return "-"
        d = metrics.get("directions", {}).get(direction, {})
        v = d.get(name)
        if v is None:
            return "-"
        return f"{float(v):.2f}"

    return (
        f"| {label} | {f('chrf_pp')} | {f('bleu')} | {f('bertscore_f1')} | {f('comet')} |"
    )


def _rare_row(label: str, rare: dict | None, direction: str) -> str:
    def f_chrf() -> str:
        if rare is None:
            return "-"
        body = rare.get("directions", {}).get(direction, {})
        bucket = body.get("buckets", {}).get("20pct_or_more", {})
        v = bucket.get("chrf_pp")
        return "-" if v is None else f"{float(v):.2f}"

    def f_oov(key: str = "recovery_rate") -> str:
        if rare is None:
            return "-"
        body = rare.get("directions", {}).get(direction, {})
        v = body.get("oov", {}).get(key)
        if v is None:
            return "-"
        if key == "recovery_rate":
            return f"{float(v):.3f}"
        return f"{int(v)}"

    return (
        f"| {label} | {f_chrf()} | "
        f"{f_oov('recovery_rate')} ({f_oov('recovered')}/{f_oov('total')}) |"
    )


def _avg_row(label: str, metrics: dict | None, rare: dict | None = None) -> str:
    def f(name: str) -> str:
        if metrics is None:
            return "-"
        v = metrics.get(name)
        if v is None:
            d = metrics.get("directions", {})
            vals = [d[k].get(name.replace("avg_", "")) for k in d if d[k].get(name.replace("avg_", "")) is not None]
            if not vals:
                return "-"
            return f"{sum(vals)/len(vals):.2f}"
        return f"{float(v):.2f}"

    rare_chrf = "-"
    rare_oov = "-"
    if rare is not None:
        v = rare.get("avg_rare_chrf_pp_20pct_or_more")
        if v is not None:
            rare_chrf = f"{float(v):.2f}"
        v = rare.get("avg_oov_recovery_rate")
        if v is not None:
            rare_oov = f"{float(v):.3f}"
    return (
        f"| {label} | {f('avg_chrf_pp')} | {f('avg_bleu')} | {rare_chrf} | {rare_oov} |"
    )


def main() -> int:
    args = parse_args()
    eval_dir = PROJECT_ROOT / "reports" / "05_nmt" / "evaluation"
    rerank_dir = PROJECT_ROOT / "reports" / "05_nmt" / "reranking"

    v0_eval = _read_metric_json(eval_dir / args.v0 / f"{args.split}_metrics.json")
    v0_rerank = _read_metric_json(rerank_dir / args.v0 / f"{args.split}_metrics_reranked.json")
    v1_eval = _read_metric_json(eval_dir / args.v1 / f"{args.split}_metrics.json")
    v1_rerank = _read_metric_json(rerank_dir / args.v1 / f"{args.split}_metrics_reranked.json")
    v0_rare = _read_metric_json(eval_dir / args.v0 / "rare_token_analysis.json")
    v0_rare_rerank = _read_metric_json(eval_dir / args.v0 / "rare_token_analysis_reranked.json")
    v1_rare = _read_metric_json(eval_dir / args.v1 / "rare_token_analysis.json")
    v1_rare_rerank = _read_metric_json(eval_dir / args.v1 / "rare_token_analysis_reranked.json")

    out_path = eval_dir / f"comparison_{args.v0}_vs_{args.v1}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append(f"# SA-BiNLLB run comparison ({args.v0} vs {args.v1})")
    sections.append("")
    sections.append(f"- Split: `{args.split}`")
    sections.append(f"- Timestamp UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    sections.append("- Headline metric: chrF++ (per plan section 31)")
    sections.append("")

    for direction, label in (("shw2spa", "Shiwilu -> Spanish"), ("spa2shw", "Spanish -> Shiwilu")):
        sections.append(f"## {label} (`{direction}`)")
        sections.append("")
        sections.append("| Variant | chrF++ | BLEU | BERTScore F1 | COMET |")
        sections.append("|---|---|---|---|---|")
        sections.append(_row(args.v0, v0_eval, direction))
        sections.append(_row(f"{args.v0} + reranker", v0_rerank, direction))
        sections.append(_row(args.v1, v1_eval, direction))
        sections.append(_row(f"{args.v1} + reranker", v1_rerank, direction))
        sections.append("")
        sections.append("### Rare-token / OOV breakdown (>=20% rare bucket)")
        sections.append("")
        sections.append("| Variant | chrF++ rare | OOV recovery (rec/tot) |")
        sections.append("|---|---|---|")
        sections.append(_rare_row(args.v0, v0_rare, direction))
        sections.append(_rare_row(f"{args.v0} + reranker", v0_rare_rerank, direction))
        sections.append(_rare_row(args.v1, v1_rare, direction))
        sections.append(_rare_row(f"{args.v1} + reranker", v1_rare_rerank, direction))
        sections.append("")

    sections.append("## Direction-averaged headline numbers")
    sections.append("")
    sections.append("| Variant | avg chrF++ | avg BLEU | avg chrF++ rare | avg OOV recovery |")
    sections.append("|---|---|---|---|---|")
    sections.append(_avg_row(args.v0, v0_eval, v0_rare))
    sections.append(_avg_row(f"{args.v0} + reranker", v0_rerank, v0_rare_rerank))
    sections.append(_avg_row(args.v1, v1_eval, v1_rare))
    sections.append(_avg_row(f"{args.v1} + reranker", v1_rerank, v1_rare_rerank))
    sections.append("")

    sections.append("## Caveats")
    sections.append("")
    sections.append(
        "- BERTScore is computed with `xlm-roberta-large`. Shiwilu is OOD for the "
        "underlying encoder; treat Shiwilu-side BERTScore as proxy."
    )
    sections.append(
        "- COMET (`Unbabel/wmt22-comet-da`) was not trained on Shiwilu data; "
        "report as indicative, not absolute."
    )
    sections.append("- chrF++ remains the primary metric for low-resource, morphology-rich languages.")
    sections.append("")
    sections.append(
        "## Source files"
        "\n- `reports/05_nmt/evaluation/<run>/<split>_metrics.json`"
        "\n- `reports/05_nmt/reranking/<run>/<split>_metrics_reranked.json`"
    )

    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[phase8a] wrote {out_path.relative_to(PROJECT_ROOT)}")

    available = {
        f"{args.v0}_eval": v0_eval is not None,
        f"{args.v0}_rerank": v0_rerank is not None,
        f"{args.v0}_rare": v0_rare is not None,
        f"{args.v0}_rare_rerank": v0_rare_rerank is not None,
        f"{args.v1}_eval": v1_eval is not None,
        f"{args.v1}_rerank": v1_rerank is not None,
        f"{args.v1}_rare": v1_rare is not None,
        f"{args.v1}_rare_rerank": v1_rare_rerank is not None,
    }
    print(f"[phase8a] available metric sources: {available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
