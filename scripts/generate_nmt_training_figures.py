"""Generate thesis figures from NMT training logs.

The figures are intentionally built from recorded training_log.json files.
They summarize optimization behavior without inventing missing histories.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "training_xl"
OUT_DIR = PROJECT_ROOT / "thesis" / "latex" / "figuras" / "generated"

SELECTED_RUN = "nllb_bidi_lora_v2_1b_loraplus_xl"

VARIANTS = {
    "v0": "nllb_bidi_lora_v0_xl",
    "v1bt": "nllb_bidi_lora_v1_bt_xl",
    "v2.1b LoRA+": "nllb_bidi_lora_v2_1b_loraplus_xl",
}


def _read_log(run_name: str) -> list[dict[str, Any]]:
    path = REPORTS_DIR / run_name / "training_log.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing training log: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _training_series(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in records:
        if "loss" in row and "epoch" in row:
            xs.append(float(row["epoch"]))
            ys.append(float(row["loss"]))
    return xs, ys


def _eval_series(records: list[dict[str, Any]], metric: str) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[float, dict[str, float]] = defaultdict(dict)
    for row in records:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        epoch_f = round(float(epoch), 6)
        for direction in ("shw2spa", "spa2shw"):
            key = f"eval_{direction}_{metric}"
            if key in row:
                grouped[epoch_f][direction] = float(row[key])

    out: dict[str, list[tuple[float, float]]] = {"shw2spa": [], "spa2shw": [], "avg": []}
    for epoch, values in sorted(grouped.items()):
        if "shw2spa" in values:
            out["shw2spa"].append((epoch, values["shw2spa"]))
        if "spa2shw" in values:
            out["spa2shw"].append((epoch, values["spa2shw"]))
        if "shw2spa" in values and "spa2shw" in values:
            out["avg"].append((epoch, (values["shw2spa"] + values["spa2shw"]) / 2.0))
    return out


def _plot_line(ax: plt.Axes, points: list[tuple[float, float]], label: str, **kwargs: Any) -> None:
    if not points:
        return
    xs, ys = zip(*points)
    ax.plot(xs, ys, marker="o", linewidth=1.9, markersize=4, label=label, **kwargs)


def figure_selected_run() -> None:
    records = _read_log(SELECTED_RUN)
    train_x, train_y = _training_series(records)
    eval_loss = _eval_series(records, "loss")
    eval_chrf = _eval_series(records, "chrf")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)

    axes[0].plot(train_x, train_y, color="#2f6f8f", linewidth=1.6)
    axes[0].set_ylabel("Loss entrenamiento")
    axes[0].set_title("Evolucion del entrenamiento de v2.1b LoRA+")

    _plot_line(axes[1], eval_loss["shw2spa"], "shw->spa", color="#c7522a")
    _plot_line(axes[1], eval_loss["spa2shw"], "spa->shw", color="#6a994e")
    axes[1].set_ylabel("Loss validacion")
    axes[1].legend(frameon=False, ncol=2, loc="best")

    _plot_line(axes[2], eval_chrf["shw2spa"], "shw->spa", color="#c7522a")
    _plot_line(axes[2], eval_chrf["spa2shw"], "spa->shw", color="#6a994e")
    _plot_line(axes[2], eval_chrf["avg"], "promedio", color="#2f2f2f", linestyle="--")
    axes[2].set_ylabel("chrF++ validacion")
    axes[2].set_xlabel("Epoca")
    axes[2].legend(frameon=False, ncol=3, loc="best")

    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"nmt_training_curve_v21b_xl.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_variant_comparison() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    colors = ["#2f6f8f", "#c7522a", "#2f2f2f"]

    for (label, run_name), color in zip(VARIANTS.items(), colors):
        records = _read_log(run_name)
        eval_chrf = _eval_series(records, "chrf")
        _plot_line(ax, eval_chrf["avg"], label, color=color)

    ax.set_title("chrF++ de validacion durante el entrenamiento")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("chrF++ promedio")
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"nmt_training_chrf_variants_xl.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global OUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    figure_selected_run()
    figure_variant_comparison()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
