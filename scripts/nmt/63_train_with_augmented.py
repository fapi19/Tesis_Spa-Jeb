"""Phase 7d runner: train v1_bt with parallel + backtranslation + mined data.

Concatenates 06_nmt_filtered/train.csv with any augmented CSVs that exist:
    07_nmt_augmented/train_bt.csv         (Phase 7a)
    07_nmt_augmented/train_mined.csv      (Phase 7b)
    07_nmt_augmented/train_morph.csv      (only if Phase 7c was forced)

Wraps the same training entrypoint as Phase 4 with a different output_dir.
Also enables Enhancement #4: per-row loss weighting (real parallel pairs
keep weight 1.0; synthetic ones are downweighted) and a bigger LoRA
(`r=32, alpha=64`) to absorb the ampler dataset without saturating.

Usage:
    python scripts/nmt/63_train_with_augmented.py \
        --config config/nmt/training.yaml \
        --output models/nmt/nllb_bidi_lora_v1_bt
    python scripts/nmt/63_train_with_augmented.py --dry-run
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

from src.nmt.training.dataset import DEFAULT_WEIGHT_MAP  # noqa: E402
from src.nmt.training.train_lora import LoraHyperparams, TrainingConfig, build_trainer  # noqa: E402

AUGMENTED_DIR = PROJECT_ROOT / "data" / "processed" / "07_nmt_augmented"

V1_BT_LORA = LoraHyperparams(
    r=32,
    alpha=64,
    dropout=0.05,
    bias="none",
    target_modules=("q_proj", "v_proj"),
    task_type="SEQ_2_SEQ_LM",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/nmt/training.yaml", type=str)
    p.add_argument("--output", default="models/nmt/nllb_bidi_lora_v1_bt", type=str)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip",
        action="append",
        choices=["bt", "mined", "morph"],
        default=[],
        help="Skip specific augmentation source(s). Repeatable.",
    )
    p.add_argument(
        "--include-morph",
        action="store_true",
        help="Include train_morph.csv if it exists (NOT recommended without linguist supervision).",
    )
    p.add_argument(
        "--no-weighting",
        action="store_true",
        help="Disable Enhancement #4 weighting (use uniform weights). Default is to weight.",
    )
    p.add_argument(
        "--no-lora-bump",
        action="store_true",
        help="Keep the v0 LoRA hyperparameters (r=16, alpha=32). Default is the v1_bt bump (r=32, alpha=64).",
    )
    p.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="ORIGIN=VALUE",
        help="Override one weight (e.g. --weight backtranslation_v0=0.4). Repeatable.",
    )
    return p.parse_args()


def _augmented_csvs(skip: list[str], include_morph: bool) -> list[Path]:
    out: list[Path] = []
    for name, key, opt in (
        ("train_bt.csv", "bt", True),
        ("train_mined.csv", "mined", True),
        ("train_morph.csv", "morph", include_morph),
    ):
        if not opt or key in skip:
            continue
        p = AUGMENTED_DIR / name
        if p.exists():
            out.append(p)
    return out


def _parse_weight_overrides(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in items:
        if "=" not in spec:
            raise SystemExit(f"--weight expects ORIGIN=VALUE, got {spec!r}")
        key, value = spec.split("=", 1)
        out[key.strip()] = float(value)
    return out


def main() -> int:
    args = parse_args()
    cfg = TrainingConfig.from_yaml(PROJECT_ROOT / args.config, PROJECT_ROOT)
    object.__setattr__(cfg.training, "output_dir", args.output)
    if not args.no_lora_bump:
        object.__setattr__(cfg, "lora", V1_BT_LORA)

    weight_map: dict[str, float] | None = None
    if not args.no_weighting:
        weight_map = dict(DEFAULT_WEIGHT_MAP)
        weight_map.update(_parse_weight_overrides(args.weight))

    extra_csvs = _augmented_csvs(args.skip, args.include_morph)
    print(f"[phase7d] augmented CSVs: {[str(p.relative_to(PROJECT_ROOT)) for p in extra_csvs] or '<none>'}")
    if weight_map is not None:
        print(f"[phase7d] weighting (Enhancement #4): {weight_map}")
    else:
        print("[phase7d] weighting disabled (--no-weighting)")
    print(f"[phase7d] LoRA: r={cfg.lora.r}, alpha={cfg.lora.alpha}, dropout={cfg.lora.dropout}")

    run_name = Path(args.output).name
    report_dir = PROJECT_ROOT / "reports" / "05_nmt" / "training" / run_name
    report_dir.mkdir(parents=True, exist_ok=True)

    trainer, info = build_trainer(
        cfg,
        project_root=PROJECT_ROOT,
        extra_train_csvs=extra_csvs,
        weight_map=weight_map,
    )
    info["augmented_csvs"] = [str(p.relative_to(PROJECT_ROOT)) for p in extra_csvs]
    print(f"[phase7d] train rows: {info['train_rows']}")
    print(f"[phase7d] validation rows by direction: {info['validation_rows_by_direction']}")
    print(f"[phase7d] LoRA r={info['lora_r']} alpha={info['lora_alpha']} fp16={info['fp16']}")
    if info.get("weighting") is not None:
        w = info["weighting"]
        print(
            f"[phase7d] weight stats: n={w['n_rows']}, mean={w['mean']:.3f}, "
            f"min={w['min']:.3f}, max={w['max']:.3f}"
        )

    if args.dry_run:
        print("[phase7d] dry-run: skipping trainer.train()")
        bootstrap = {
            "phase": "7d",
            "run_name": run_name,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "dry_run",
            "info": info,
        }
        (report_dir / "bootstrap.json").write_text(
            json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    print(f"[phase7d] calling trainer.train() (output_dir={args.output})")
    train_result = trainer.train()

    print("[phase7d] saving best LoRA adapter + tokenizer")
    trainer.save_model(args.output)
    trainer.processing_class.save_pretrained(args.output)

    summary = {
        "phase": "7d",
        "run_name": run_name,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "info": info,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
        "num_train_epochs": trainer.state.epoch,
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "metric_for_best_model": cfg.training.metric_for_best_model,
        "fp16": info["fp16"],
        "extra_train_csvs": [str(p.relative_to(PROJECT_ROOT)) for p in extra_csvs],
    }
    (report_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[phase7d] wrote {report_dir.relative_to(PROJECT_ROOT)}/{{training_log,summary}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
