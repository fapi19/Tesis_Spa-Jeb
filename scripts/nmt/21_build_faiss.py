"""Phase 2b runner: build FAISS IndexFlatIP for shiwilu and spanish sides
on the accepted-train embeddings.

Outputs:
    data/processed/06_nmt_filtered/faiss_shw.index
    data/processed/06_nmt_filtered/faiss_shw_meta.parquet
    data/processed/06_nmt_filtered/faiss_spa.index
    data/processed/06_nmt_filtered/faiss_spa_meta.parquet
    reports/05_nmt/preprocessing/faiss_index.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nmt.preprocessing.faiss_index import (  # noqa: E402
    build_side_index,
    collect_pairs_for_indexing,
    near_duplicate_check,
)
from src.nmt.preprocessing.semantic_filter import (  # noqa: E402
    SemanticFilterConfig,
    load_embedding_model,
    resolve_device,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "nmt" / "filter.yaml"
FILTERED_DIR = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered"
REPORTS_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "preprocessing"


def main() -> int:
    cfg = SemanticFilterConfig.from_yaml(CONFIG_PATH, PROJECT_ROOT)
    device = resolve_device(cfg.device)
    print(f"[phase2b] device={device}")

    pairs = collect_pairs_for_indexing(FILTERED_DIR)
    print(f"[phase2b] indexing {len(pairs)} accepted pairs")

    model = load_embedding_model(cfg)

    artifacts = []
    for side in ("shiwilu", "spanish"):
        artifact = build_side_index(
            pairs,
            side,
            model,
            FILTERED_DIR,
            batch_size=cfg.batch_size,
            use_e5=cfg.use_e5_prefixes,
        )
        artifacts.append(artifact)
        print(
            f"[phase2b] {side}: index={artifact.index_path.name} "
            f"meta={artifact.meta_path.name} n={artifact.n_vectors} dim={artifact.dim}"
        )

    near_dup_reports: list[dict] = []
    for artifact in artifacts:
        rep = near_duplicate_check(artifact, pairs, ip_threshold=0.98)
        print(f"[phase2b] {artifact.side}: near-duplicate pairs (IP>=0.98) = {rep['flagged_count']}")
        near_dup_reports.append(rep)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "2b",
        "step": "build_faiss",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "model_path": str(cfg.model_path.relative_to(PROJECT_ROOT)),
            "use_e5_prefixes": cfg.use_e5_prefixes,
            "batch_size": cfg.batch_size,
            "device": device,
        },
        "indices": [
            {
                "side": a.side,
                "index_path": str(a.index_path.relative_to(PROJECT_ROOT)),
                "meta_path": str(a.meta_path.relative_to(PROJECT_ROOT)),
                "n_vectors": a.n_vectors,
                "dim": a.dim,
            }
            for a in artifacts
        ],
        "near_duplicate_scan": [
            {
                "side": rep["side"],
                "ip_threshold": rep["ip_threshold"],
                "flagged_count": rep["flagged_count"],
                "total_vectors": rep["total_vectors"],
                "examples": rep["flagged_pairs"][:20],
            }
            for rep in near_dup_reports
        ],
    }
    out_path = REPORTS_DIR / "faiss_index.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase2b] report -> {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
