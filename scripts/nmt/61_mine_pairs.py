"""Phase 7b runner: mine cross-lingual pairs in the v3 SBERT space.

Outputs:
    data/processed/07_nmt_augmented/train_mined.csv
    reports/05_nmt/augmentation/mining.json

Modes:
    --internal (default): query the parallel corpus against itself to find
        non-trivial nearest neighbors that pass reciprocal-NN + IP > min_ip.
    --extra-spa-text / --extra-shw-text: external monolingual mining (use
        when external Spanish/Shiwilu text becomes available).
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

import pandas as pd  # noqa: E402

from scripts.nmt._paths import resolve_paths  # noqa: E402
from src.nmt.augmentation.embedding_mining import (  # noqa: E402
    MiningConfig,
    mine_external,
    mine_internal,
    to_canonical_dataframe,
)
from src.nmt.preprocessing.semantic_filter import (  # noqa: E402
    SemanticFilterConfig,
    load_embedding_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="main")
    p.add_argument("--min-ip", type=float, default=0.65)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--no-reciprocal",
        action="store_true",
        help="Disable reciprocal-NN constraint (internal mining only).",
    )
    p.add_argument("--extra-spa-text", action="append", default=[], type=str)
    p.add_argument("--extra-shw-text", action="append", default=[], type=str)
    return p.parse_args()


def _read_lines(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
    return out


def main() -> int:
    args = parse_args()
    nmt = resolve_paths(PROJECT_ROOT, args.variant)
    filtered_dir = nmt.filtered_dir
    augmented_dir = nmt.augmented_dir
    suffix = "_xl" if args.variant == "xl" else ""
    reports_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"augmentation{suffix}"

    cfg = MiningConfig(min_ip=args.min_ip, top_k=args.top_k, require_reciprocal=not args.no_reciprocal)
    filter_cfg = SemanticFilterConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "filter.yaml", PROJECT_ROOT)
    print(f"[phase7b] variant={args.variant} min_ip={cfg.min_ip} top_k={cfg.top_k} reciprocal={cfg.require_reciprocal}")
    print(f"[phase7b] filtered_dir={filtered_dir.relative_to(PROJECT_ROOT)}")

    augmented_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    extra_spa = _read_lines([Path(p).resolve() for p in args.extra_spa_text])
    extra_shw = _read_lines([Path(p).resolve() for p in args.extra_shw_text])

    mined_internal_df, info_internal = mine_internal(filtered_dir, cfg, filter_cfg)
    print(f"[phase7b] internal candidates: {info_internal['raw_candidates']}")

    info_external = None
    mined_external_df = pd.DataFrame()
    if extra_spa or extra_shw:
        sbert = load_embedding_model(filter_cfg)
        mined_external_df, info_external = mine_external(
            filtered_dir, extra_spa, extra_shw, cfg, sbert, filter_cfg
        )
        print(f"[phase7b] external candidates: {info_external['raw_candidates']}")

    # Combine both sources before converting to canonical schema.
    cols = ["spanish", "shiwilu", "ip", "rank"]
    parts = []
    if not mined_internal_df.empty:
        parts.append(mined_internal_df[cols].copy())
    if not mined_external_df.empty:
        parts.append(mined_external_df[cols].copy())
    if parts:
        merged = pd.concat(parts, ignore_index=True)
        merged = (
            merged.sort_values("ip", ascending=False)
            .drop_duplicates(subset=["spanish", "shiwilu"])
            .reset_index(drop=True)
        )
    else:
        merged = pd.DataFrame(columns=cols)

    canonical = to_canonical_dataframe(merged)
    out_csv = augmented_dir / "train_mined.csv"
    canonical.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[phase7b] wrote {out_csv.relative_to(PROJECT_ROOT)} ({len(canonical)} rows from {len(merged)} pairs)")

    report = {
        "phase": "7b",
        "step": "mine_pairs",
        "variant": args.variant,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "internal": info_internal,
        "external": info_external,
        "after_dedup_pairs": int(len(merged)),
        "directional_rows": int(len(canonical)),
        "ip_stats": {
            "mean": float(merged["ip"].mean()) if len(merged) else None,
            "min": float(merged["ip"].min()) if len(merged) else None,
            "max": float(merged["ip"].max()) if len(merged) else None,
        },
    }
    out_path = reports_dir / "mining.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase7b] report -> {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
