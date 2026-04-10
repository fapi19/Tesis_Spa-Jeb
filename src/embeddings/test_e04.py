from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from .run_experiment import EXPERIMENTS, prepare_shared_data, run_experiment


SEEDS = [42, 123, 777]
EXPERIMENT_KEYS = ["E0", "E4"]


def load_metrics(checkpoint_path: Path) -> dict[str, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return {
        "best_r1": float(ckpt.get("best_r1", 0.0)),
        "best_r5": float(ckpt.get("best_r5", 0.0)),
        "best_mrr": float(ckpt.get("best_mrr", 0.0)),
        "val_loss": float(ckpt.get("val_loss", 0.0)),
    }


def summarize(metric_values: list[float]) -> tuple[float, float]:
    avg = mean(metric_values)
    std = stdev(metric_values) if len(metric_values) > 1 else 0.0
    return avg, std


def main() -> None:
    print("=" * 60)
    print("Paso 0: preparar datos compartidos")
    print("=" * 60)
    prepare_shared_data()

    all_results: dict[str, list[dict[str, Any]]] = {k: [] for k in EXPERIMENT_KEYS}

    for exp_key in EXPERIMENT_KEYS:
        base_cfg = EXPERIMENTS[exp_key]

        print()
        print("=" * 60)
        print(f"Experimento base: {exp_key} -> {base_cfg.name}")
        print("=" * 60)

        for seed in SEEDS:
            seeded_cfg = replace(
                base_cfg,
                seed=seed,
                name=f"{base_cfg.name}_seed{seed}",
                description=f"{base_cfg.description} | seed={seed}",
            )

            print()
            print("-" * 60)
            print(f"Corriendo {exp_key} con seed={seed}")
            print("-" * 60)

            checkpoint_path = run_experiment(seeded_cfg)
            metrics = load_metrics(Path(checkpoint_path))

            result = {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                **metrics,
            }
            all_results[exp_key].append(result)

    print()
    print("=" * 60)
    print("Resumen final")
    print("=" * 60)

    for exp_key, rows in all_results.items():
        r1s = [r["best_r1"] for r in rows]
        r5s = [r["best_r5"] for r in rows]
        mrrs = [r["best_mrr"] for r in rows]
        vls = [r["val_loss"] for r in rows]

        r1_avg, r1_std = summarize(r1s)
        r5_avg, r5_std = summarize(r5s)
        mrr_avg, mrr_std = summarize(mrrs)
        vl_avg, vl_std = summarize(vls)

        print()
        print(f"{exp_key}")
        for row in rows:
            print(
                f"  seed={row['seed']} | "
                f"R@1={row['best_r1']:.4f} | "
                f"R@5={row['best_r5']:.4f} | "
                f"MRR={row['best_mrr']:.4f} | "
                f"val_loss={row['val_loss']:.4f}"
            )

        print(
            f"  PROMEDIO -> "
            f"R@1={r1_avg:.4f} ± {r1_std:.4f} | "
            f"R@5={r5_avg:.4f} ± {r5_std:.4f} | "
            f"MRR={mrr_avg:.4f} ± {mrr_std:.4f} | "
            f"val_loss={vl_avg:.4f} ± {vl_std:.4f}"
        )


if __name__ == "__main__":
    main()