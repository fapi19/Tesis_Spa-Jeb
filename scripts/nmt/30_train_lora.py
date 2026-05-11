"""Phase 4 runner: NLLB + LoRA bidirectional fine-tuning.

Trains the v0 adapter on data/processed/06_nmt_filtered/{train,valid}.csv,
saving the LoRA adapter + extended tokenizer under
models/nmt/nllb_bidi_lora_v0/ (or whatever output_dir the YAML specifies).

Usage:
    python scripts/nmt/30_train_lora.py --config config/nmt/training.yaml
    python scripts/nmt/30_train_lora.py --config config/nmt/training.yaml --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nmt.training.train_lora import TrainingConfig, build_trainer  # noqa: E402
from scripts.nmt._paths import resolve_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/nmt/training.yaml", type=str)
    p.add_argument("--variant", choices=["main", "xl"], default="main")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build trainer + datasets, print sanity info, do not call train().",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps (used for short smoke trainings).",
    )
    p.add_argument(
        "--report",
        type=str,
        default=None,
        help="Override training report directory (default: reports/05_nmt/training/<run>/).",
    )
    # Phase 0 ablation flags. Defaults preserve existing v0/v1 behaviour.
    p.add_argument("--use-dora", action="store_true",
                   help="Use DoRA (Decomposed LoRA) instead of LoRA. Better for low-resource.")
    p.add_argument("--loraplus-lr-ratio", type=float, default=0.0,
                   help="LoRA+ asymmetric LR ratio (lr_B/lr_A). 16.0 is paper default. 0 disables.")
    p.add_argument("--rank", type=int, default=None,
                   help="Override LoRA r (default from yaml).")
    p.add_argument("--alpha", type=int, default=None,
                   help="Override LoRA alpha (default from yaml).")
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 instead of fp16 (more numerically stable on Blackwell).")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="Apply torch.compile to the model (PyTorch 2.x graph optimisation).")
    p.add_argument("--direction", choices=["shw2spa", "spa2shw"], default=None,
                   help="Train only this direction. Used for Two-DoRA training (one adapter per direction).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output_dir (default from yaml + variant convention).")
    p.add_argument("--run-suffix", type=str, default=None,
                   help="Append suffix to default run name (e.g. _dora, _loraplus).")
    return p.parse_args()


def _resolve_run_name(cfg: TrainingConfig) -> str:
    out = Path(cfg.training.output_dir)
    return out.name


def main() -> int:
    args = parse_args()
    nmt_paths = resolve_paths(PROJECT_ROOT, args.variant)
    cfg = TrainingConfig.from_yaml(PROJECT_ROOT / args.config, PROJECT_ROOT)

    # Variant routing: rewrite data CSVs + default output dir for xl.
    if args.variant == "xl":
        data_cfg = replace(
            cfg.data,
            train_csv=nmt_paths.filtered_dir / "train.csv",
            valid_csv=nmt_paths.filtered_dir / "valid.csv",
            test_csv=nmt_paths.filtered_dir / "test.csv",
        )
        default_output = str(PROJECT_ROOT / "models" / "nmt" / "nllb_bidi_lora_v0_xl")
        training_cfg = replace(cfg.training, output_dir=default_output)
        cfg = replace(cfg, data=data_cfg, training=training_cfg)

    # Apply Phase 0 CLI overrides.
    lora_overrides: dict = {}
    if args.use_dora:
        lora_overrides["use_dora"] = True
    if args.loraplus_lr_ratio > 0:
        lora_overrides["loraplus_lr_ratio"] = args.loraplus_lr_ratio
    if args.rank is not None:
        lora_overrides["r"] = args.rank
    if args.alpha is not None:
        lora_overrides["alpha"] = args.alpha
    if lora_overrides:
        cfg = replace(cfg, lora=replace(cfg.lora, **lora_overrides))

    training_overrides: dict = {}
    if args.bf16:
        training_overrides["precision"] = "bf16"
    if args.compile_model:
        training_overrides["compile_model"] = True
    if args.output_dir is not None:
        training_overrides["output_dir"] = str(args.output_dir)
    elif args.run_suffix:
        base = Path(cfg.training.output_dir).name
        training_overrides["output_dir"] = str(
            PROJECT_ROOT / "models" / "nmt" / f"{base}{args.run_suffix}"
        )
    if training_overrides:
        cfg = replace(cfg, training=replace(cfg.training, **training_overrides))

    run_name = _resolve_run_name(cfg)
    report_dir = (
        Path(args.report) if args.report else nmt_paths.reports_training_dir / run_name
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase4] variant={args.variant}, run_name={run_name}")
    print(f"[phase4] output_dir={cfg.training.output_dir}")
    print(f"[phase4] report_dir={report_dir.relative_to(PROJECT_ROOT)}")
    print(f"[phase4] LoRA: r={cfg.lora.r} alpha={cfg.lora.alpha} dora={cfg.lora.use_dora} loraplus_ratio={cfg.lora.loraplus_lr_ratio}")
    print(f"[phase4] precision={cfg.training.precision} compile={cfg.training.compile_model} direction_filter={args.direction}")

    trainer, info = build_trainer(cfg, project_root=PROJECT_ROOT, direction_filter=args.direction)

    print(f"[phase4] train rows: {info['train_rows']}")
    print(f"[phase4] validation rows by direction: {info['validation_rows_by_direction']}")
    print(f"[phase4] LoRA r={info['lora_r']} alpha={info['lora_alpha']} dora={info['use_dora']} fp16={info['fp16']} bf16={info['bf16']}")

    if args.max_steps is not None:
        trainer.args.max_steps = args.max_steps
        trainer.args.num_train_epochs = 0
        print(f"[phase4] OVERRIDE max_steps={args.max_steps}")

    if args.dry_run:
        print("[phase4] dry-run: skipping trainer.train()")
        bootstrap = {
            "phase": 4,
            "run_name": run_name,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "dry_run",
            "info": info,
            "training_args": trainer.args.to_dict(),
        }
        (report_dir / "bootstrap.json").write_text(
            json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    print("[phase4] calling trainer.train() (this is the long-running step)")
    train_result = trainer.train()

    print("[phase4] saving best LoRA adapter + tokenizer")
    trainer.save_model(cfg.training.output_dir)
    trainer.processing_class.save_pretrained(cfg.training.output_dir)

    log_history = trainer.state.log_history
    summary = {
        "phase": 4,
        "run_name": run_name,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "info": info,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
        "num_train_epochs": trainer.state.epoch,
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "metric_for_best_model": cfg.training.metric_for_best_model,
        "fp16": info["fp16"],
    }
    (report_dir / "training_log.json").write_text(
        json.dumps(log_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[phase4] wrote {report_dir.relative_to(PROJECT_ROOT)}/{{training_log,summary}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
