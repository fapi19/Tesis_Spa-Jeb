"""Phase 6: bootstrap confidence intervals for chrF++ on test predictions.

For each run, resamples the reranked test predictions with replacement N times
and computes 95% CI on chrF++ per direction + average.

Reads:
    reports/05_nmt/reranking_xl/<run>/test_predictions_reranked.jsonl
    (falls back to evaluation_xl/<run>/test_predictions.jsonl if reranked
    is unavailable)

Emits:
    reports/05_nmt/evaluation_xl/<run>/bootstrap_ci.json
    reports/05_nmt/evaluation_xl/bootstrap_ci_summary.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np
import sacrebleu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUNS = [
    "nllb_bidi_lora_v0_xl",
    "nllb_bidi_lora_v1_bt_xl",
    "nllb_bidi_lora_v2_0_dora_xl",
    "nllb_bidi_lora_v2_1_dora_loraplus_xl",
    "nllb_bidi_lora_v2_1b_loraplus_xl",
    "nllb_bidi_lora_v2_2_bt_iter1_xl",
]

LABELS = {
    "nllb_bidi_lora_v0_xl": "v0_xl",
    "nllb_bidi_lora_v1_bt_xl": "v1_bt_xl",
    "nllb_bidi_lora_v2_0_dora_xl": "v2.0 DoRA",
    "nllb_bidi_lora_v2_1_dora_loraplus_xl": "v2.1 DoRA+LoRA+",
    "nllb_bidi_lora_v2_1b_loraplus_xl": "v2.1b LoRA+",
    "nllb_bidi_lora_v2_2_bt_iter1_xl": "v2.2 BT iter1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="xl")
    p.add_argument("--runs", nargs="+", default=None)
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap samples (default 1000)")
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--use-baseline", action="store_true",
                   help="Use non-reranked baseline predictions instead of reranked")
    return p.parse_args()


def _suffix(variant: str) -> str:
    return "_xl" if variant == "xl" else ""


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _chrf_pp(hyps: list[str], refs: list[str]) -> float:
    """sacrebleu chrF++ (word_order=2). Matches 40_evaluate.py settings."""
    score = sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2)
    return float(score.score)


def _bootstrap_direction(hyps: list[str], refs: list[str], n_boot: int, rng: np.random.Generator) -> dict:
    n = len(hyps)
    point = _chrf_pp(hyps, refs)
    samples = np.empty(n_boot, dtype=np.float64)
    idx_all = np.arange(n)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sampled_hyps = [hyps[i] for i in idx]
        sampled_refs = [refs[i] for i in idx]
        samples[b] = _chrf_pp(sampled_hyps, sampled_refs)
    ci_lo = float(np.percentile(samples, 2.5))
    ci_hi = float(np.percentile(samples, 97.5))
    return {
        "n": n,
        "point": point,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "ci95_halfwidth": (ci_hi - ci_lo) / 2.0,
    }


def _process_run(run: str, variant: str, n_boot: int, seed: int, use_baseline: bool) -> dict:
    suffix = _suffix(variant)
    rerank_path = PROJECT_ROOT / "reports" / "05_nmt" / f"reranking{suffix}" / run / "test_predictions_reranked.jsonl"
    base_path = PROJECT_ROOT / "reports" / "05_nmt" / f"evaluation{suffix}" / run / "test_predictions.jsonl"
    if use_baseline or not rerank_path.exists():
        pred_path = base_path
        variant_tag = "baseline"
    else:
        pred_path = rerank_path
        variant_tag = "reranked"
    if not pred_path.exists():
        return {"run": run, "error": f"no predictions at {pred_path}"}

    rows = _load_jsonl(pred_path)
    by_dir: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        d = r.get("direction") or f"{r.get('source_lang')}2{r.get('target_lang')}"
        if d not in by_dir:
            by_dir[d] = {"hyps": [], "refs": []}
        by_dir[d]["hyps"].append(r.get("hypothesis", ""))
        by_dir[d]["refs"].append(r.get("reference", ""))

    rng = np.random.default_rng(seed)
    out_directions = {}
    for d, payload in by_dir.items():
        out_directions[d] = _bootstrap_direction(payload["hyps"], payload["refs"], n_boot, rng)

    if "shw2spa" in out_directions and "spa2shw" in out_directions:
        n_boot_avg = n_boot
        avg_samples = np.empty(n_boot_avg, dtype=np.float64)
        a_hyps = by_dir["shw2spa"]["hyps"]; a_refs = by_dir["shw2spa"]["refs"]
        b_hyps = by_dir["spa2shw"]["hyps"]; b_refs = by_dir["spa2shw"]["refs"]
        rng2 = np.random.default_rng(seed + 1)
        for k in range(n_boot_avg):
            ia = rng2.integers(0, len(a_hyps), len(a_hyps))
            ib = rng2.integers(0, len(b_hyps), len(b_hyps))
            sa = _chrf_pp([a_hyps[i] for i in ia], [a_refs[i] for i in ia])
            sb = _chrf_pp([b_hyps[i] for i in ib], [b_refs[i] for i in ib])
            avg_samples[k] = (sa + sb) / 2.0
        out_directions["avg"] = {
            "n": len(a_hyps) + len(b_hyps),
            "point": (out_directions["shw2spa"]["point"] + out_directions["spa2shw"]["point"]) / 2.0,
            "mean": float(avg_samples.mean()),
            "std": float(avg_samples.std(ddof=1)),
            "ci95_lo": float(np.percentile(avg_samples, 2.5)),
            "ci95_hi": float(np.percentile(avg_samples, 97.5)),
            "ci95_halfwidth": float((np.percentile(avg_samples, 97.5) - np.percentile(avg_samples, 2.5)) / 2.0),
        }

    return {
        "run": run,
        "label": LABELS.get(run, run),
        "prediction_source": variant_tag,
        "prediction_path": str(pred_path.relative_to(PROJECT_ROOT)),
        "n_boot": n_boot,
        "seed": seed,
        "directions": out_directions,
    }


def render_summary(results: list[dict], n_boot: int) -> str:
    lines = []
    lines.append(f"# Bootstrap 95% CIs on chrF++ — Phase 6")
    lines.append("")
    lines.append(f"Generated {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}. "
                 f"Bootstrap n={n_boot} resamples per run. "
                 f"Predictions are **reranked** unless flagged otherwise.")
    lines.append("")
    lines.append("| Run | shw→spa chrF++ [95% CI] | spa→shw chrF++ [95% CI] | avg chrF++ [95% CI] |")
    lines.append("|---|---|---|---|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['run']} | ERROR: {r['error']} | — | — |")
            continue
        def cell(d):
            x = r["directions"].get(d)
            if not x:
                return "—"
            return f"{x['point']:.2f} [{x['ci95_lo']:.2f}, {x['ci95_hi']:.2f}]"
        lines.append(f"| {r['label']} ({r['prediction_source']}) | {cell('shw2spa')} | {cell('spa2shw')} | {cell('avg')} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- CI = percentile bootstrap on the 446 test items per direction (892 total).")
    lines.append("- `avg` CI is computed by independently resampling each direction and averaging — it is wider than a fixed-weight transform of the per-direction CIs.")
    lines.append("- Overlapping 95% CIs between two runs *do not* prove the diff is non-significant, but non-overlapping CIs do strongly suggest a real difference.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs = args.runs or DEFAULT_RUNS
    results = []
    for run in runs:
        print(f"[boot] {run}")
        res = _process_run(run, args.variant, args.n_boot, args.seed, args.use_baseline)
        results.append(res)
        if "error" not in res:
            suffix = _suffix(args.variant)
            out_path = PROJECT_ROOT / "reports" / "05_nmt" / f"evaluation{suffix}" / run / "bootstrap_ci.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"  wrote {out_path}")
            for d in ("shw2spa", "spa2shw", "avg"):
                x = res["directions"].get(d)
                if x:
                    print(f"    {d}: chrF++={x['point']:.2f}  95% CI [{x['ci95_lo']:.2f}, {x['ci95_hi']:.2f}]")

    summary_path = PROJECT_ROOT / "reports" / "05_nmt" / f"evaluation{_suffix(args.variant)}" / "bootstrap_ci_summary.md"
    summary_path.write_text(render_summary(results, args.n_boot), encoding="utf-8")
    print(f"\n[boot] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
