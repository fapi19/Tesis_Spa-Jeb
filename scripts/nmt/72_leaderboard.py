"""Phase 5 deliverable: N-way leaderboard across all trained NMT runs.

Reads:
    reports/05_nmt/evaluation_xl/<run>/test_metrics.json
    reports/05_nmt/reranking_xl/<run>/test_metrics_reranked.json
        (also pulls best-alpha row from .../ablation.json when present)

Emits:
    reports/05_nmt/evaluation_xl/leaderboard.md
    reports/05_nmt/evaluation_xl/leaderboard.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_RUNS = [
    "nllb_bidi_lora_v0_xl",
    "nllb_bidi_lora_v1_bt_xl",
    "nllb_bidi_lora_v2_0_dora_xl",
    "nllb_bidi_lora_v2_1_dora_loraplus_xl",
    "nllb_bidi_lora_v2_1b_loraplus_xl",
    "nllb_bidi_lora_v2_2_bt_iter1_xl",
]

LABELS = {
    "nllb_bidi_lora_v0_xl": "v0_xl (baseline LoRA r=16)",
    "nllb_bidi_lora_v1_bt_xl": "v1_bt_xl (+BT OPUS-100, LoRA r=32)",
    "nllb_bidi_lora_v2_0_dora_xl": "v2.0 DoRA",
    "nllb_bidi_lora_v2_1_dora_loraplus_xl": "v2.1 DoRA + LoRA+",
    "nllb_bidi_lora_v2_1b_loraplus_xl": "v2.1b LoRA+ (champion)",
    "nllb_bidi_lora_v2_2_bt_iter1_xl": "v2.2 +BT iter1 Wikipedia (regression)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="xl")
    p.add_argument("--runs", nargs="+", default=None,
                   help="Optional override of run names (defaults to the canonical 6)")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    return p.parse_args()


def _suffix(variant: str) -> str:
    return "_xl" if variant == "xl" else ""


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(x, digits=2):
    if x is None:
        return "-"
    return f"{float(x):.{digits}f}"


def _direction(metrics: dict | None, direction: str) -> dict:
    if not metrics:
        return {}
    return metrics.get("directions", {}).get(direction, {})


def _summarize(run: str, variant: str, split: str) -> dict:
    suffix = _suffix(variant)
    eval_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"evaluation{suffix}" / run
    rerank_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"reranking{suffix}" / run

    base = _load(eval_dir / f"{split}_metrics.json") or {}
    reranked = _load(rerank_dir / f"{split}_metrics_reranked.json") or {}
    ablation = _load(rerank_dir / "ablation.json") or {}

    out = {
        "run": run,
        "label": LABELS.get(run, run),
        "baseline": {
            "shw2spa": _direction(base, "shw2spa"),
            "spa2shw": _direction(base, "spa2shw"),
            "avg_chrf_pp": base.get("avg_chrf_pp"),
            "avg_bleu": base.get("avg_bleu"),
        },
        "reranked": {
            "shw2spa": _direction(reranked, "shw2spa"),
            "spa2shw": _direction(reranked, "spa2shw"),
            "avg_chrf_pp": reranked.get("avg_chrf_pp"),
            "avg_bleu": reranked.get("avg_bleu"),
            "best_alpha": reranked.get("meta", {}).get("best_alpha"),
        },
        "ablation_best_alpha": (ablation.get("best") or {}).get("alpha"),
    }
    return out


def render_md(rows: list[dict], split: str, variant: str) -> str:
    lines = []
    lines.append(f"# NMT Leaderboard — Phase 5 (split={split}, variant={variant})")
    lines.append("")
    lines.append(f"Generated {dt.datetime.utcnow().isoformat(timespec='seconds')}Z. "
                 f"All metrics on the held-out **test** set (446 pairs × 2 directions = 892 rows).")
    lines.append("")

    lines.append("## Baseline (no reranking)")
    lines.append("")
    lines.append("| Run | shw→spa chrF++ / BLEU / BERTScore-F1 / COMET | spa→shw chrF++ / BLEU / BERTScore-F1 / COMET | avg chrF++ |")
    lines.append("|---|---|---|---:|")
    for r in rows:
        s = r["baseline"]["shw2spa"]
        t = r["baseline"]["spa2shw"]
        avg = r["baseline"]["avg_chrf_pp"]
        s_cell = f"{_fmt(s.get('chrf_pp'))} / {_fmt(s.get('bleu'))} / {_fmt(s.get('bertscore_f1'),3)} / {_fmt(s.get('comet'),3)}"
        t_cell = f"{_fmt(t.get('chrf_pp'))} / {_fmt(t.get('bleu'))} / {_fmt(t.get('bertscore_f1'),3)} / {_fmt(t.get('comet'),3)}"
        lines.append(f"| {r['label']} | {s_cell} | {t_cell} | **{_fmt(avg)}** |")
    lines.append("")

    lines.append("## Reranked (best-alpha by avg chrF++)")
    lines.append("")
    lines.append("| Run | best α | shw→spa chrF++ / BLEU | spa→shw chrF++ / BLEU | avg chrF++ | Δ vs baseline |")
    lines.append("|---|---:|---|---|---:|---:|")
    for r in rows:
        rb = r["reranked"]
        bb = r["baseline"]
        alpha = r["ablation_best_alpha"] if r["ablation_best_alpha"] is not None else rb.get("best_alpha")
        s = rb["shw2spa"]
        t = rb["spa2shw"]
        avg_r = rb["avg_chrf_pp"]
        avg_b = bb["avg_chrf_pp"]
        delta = None
        if avg_r is not None and avg_b is not None:
            delta = float(avg_r) - float(avg_b)
        s_cell = f"{_fmt(s.get('chrf_pp'))} / {_fmt(s.get('bleu'))}"
        t_cell = f"{_fmt(t.get('chrf_pp'))} / {_fmt(t.get('bleu'))}"
        delta_str = "-" if delta is None else f"{delta:+.2f}"
        alpha_str = "-" if alpha is None else f"{float(alpha):.2f}"
        lines.append(f"| {r['label']} | {alpha_str} | {s_cell} | {t_cell} | **{_fmt(avg_r)}** | {delta_str} |")
    lines.append("")

    # Champion line
    champ = max(rows, key=lambda r: (r["reranked"]["avg_chrf_pp"] or -1.0))
    lines.append("---")
    lines.append("")
    lines.append(f"**Shipped champion: `{champ['label']}`** "
                 f"with reranked avg chrF++ = **{_fmt(champ['reranked']['avg_chrf_pp'])}** "
                 f"(α = {_fmt(champ['ablation_best_alpha'] or champ['reranked'].get('best_alpha'), 2)}).")
    lines.append("")
    lines.append("Caveats (from each `test_metrics.json`):")
    lines.append("- **BERTScore** uses `xlm-roberta-large` which has multilingual coverage but Shiwilu is OOD. Treat Shiwilu-side BERTScore as proxy only.")
    lines.append("- **COMET** (`Unbabel/wmt22-comet-da`) was not trained on Shiwilu. Reported as indicative only.")
    lines.append("- Primary headline metric is **chrF++** per plan §31.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs = args.runs or DEFAULT_RUNS
    rows = [_summarize(r, args.variant, args.split) for r in runs]

    suffix = _suffix(args.variant)
    out_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"evaluation{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "leaderboard.md"
    json_path = out_dir / "leaderboard.json"
    md_path.write_text(render_md(rows, args.split, args.variant), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "variant": args.variant,
                "generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
