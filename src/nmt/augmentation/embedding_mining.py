"""Phase 7b: bilingual mining via FAISS in the v3 SBERT space.

Mines candidate cross-lingual paraphrase pairs that the parallel corpus
doesn't already contain. Two modes:

    1. Internal (default): uses the parallel-corpus FAISS indices both as
       queries and as the search pool. For each Spanish row, find the top-K
       nearest Shiwilu rows; accept (Spanish_q, Shiwilu_n) only if
       n != q (different pair) AND the relationship is reciprocal-NN AND
       IP > min_ip.

    2. External pool: optional --extra-shw-text / --extra-spa-text files
       enable mining against external monolingual data once available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..preprocessing.semantic_filter import (
    SemanticFilterConfig,
    encode,
    load_embedding_model,
    resolve_device,
)


@dataclass(frozen=True)
class MiningConfig:
    min_ip: float = 0.65
    top_k: int = 5
    require_reciprocal: bool = True


def _load_index_and_meta(idx_path: Path, meta_path: Path) -> tuple[faiss.Index, pd.DataFrame]:
    return faiss.read_index(str(idx_path)), pd.read_parquet(meta_path)


def mine_internal(
    filtered_dir: Path,
    cfg: MiningConfig,
    embedding_cfg: SemanticFilterConfig,
) -> tuple[pd.DataFrame, dict]:
    """Mine candidate pairs using only the internal parallel-corpus indices."""
    shw_idx, shw_meta = _load_index_and_meta(
        filtered_dir / "faiss_shw.index",
        filtered_dir / "faiss_shw_meta.parquet",
    )
    spa_idx, spa_meta = _load_index_and_meta(
        filtered_dir / "faiss_spa.index",
        filtered_dir / "faiss_spa_meta.parquet",
    )

    if shw_meta["pair_id"].tolist() != spa_meta["pair_id"].tolist():
        raise RuntimeError("shw and spa FAISS metadata are not row-aligned by pair_id")

    n = shw_idx.ntotal
    shw_vecs = np.vstack([shw_idx.reconstruct(i) for i in range(n)]).astype(np.float32)
    spa_vecs = np.vstack([spa_idx.reconstruct(i) for i in range(n)]).astype(np.float32)

    # For each Spanish row, find top-K Shiwilu neighbors.
    sims_spa2shw, idx_spa2shw = shw_idx.search(spa_vecs, cfg.top_k)
    # For each Shiwilu row, find top-K Spanish neighbors.
    sims_shw2spa, idx_shw2spa = spa_idx.search(shw_vecs, cfg.top_k)

    candidates: list[dict] = []
    for q in range(n):
        for rank, (n_idx, n_sim) in enumerate(zip(idx_spa2shw[q], sims_spa2shw[q])):
            n_idx = int(n_idx)
            if n_idx < 0 or n_idx == q:
                continue
            if float(n_sim) < cfg.min_ip:
                continue
            if cfg.require_reciprocal:
                # nearest Spanish to shw[n_idx] should include q
                nn_back = idx_shw2spa[n_idx].tolist()
                if q not in nn_back:
                    continue
            candidates.append(
                {
                    "spa_row": q,
                    "shw_row": n_idx,
                    "spa_pair_id": spa_meta["pair_id"].iloc[q],
                    "shw_pair_id": shw_meta["pair_id"].iloc[n_idx],
                    "spanish": spa_meta["spanish"].iloc[q],
                    "shiwilu": shw_meta["shiwilu"].iloc[n_idx],
                    "ip": float(n_sim),
                    "rank": int(rank),
                }
            )

    df = pd.DataFrame(candidates)
    if not df.empty:
        df = df.sort_values("ip", ascending=False).drop_duplicates(
            subset=["spa_row", "shw_row"]
        ).reset_index(drop=True)

    info = {
        "mode": "internal",
        "min_ip": cfg.min_ip,
        "top_k": cfg.top_k,
        "require_reciprocal": cfg.require_reciprocal,
        "candidate_pool_size": int(n),
        "raw_candidates": int(len(df)),
    }
    return df, info


def mine_external(
    filtered_dir: Path,
    extra_spa_text: Sequence[str],
    extra_shw_text: Sequence[str],
    cfg: MiningConfig,
    model: SentenceTransformer,
    embedding_cfg: SemanticFilterConfig,
) -> tuple[pd.DataFrame, dict]:
    """Mine candidates by using external monolingual queries against the
    parallel-corpus FAISS indices.

    For each Spanish query: find top-K Shiwilu neighbors in shw_idx.
    For each Shiwilu query: find top-K Spanish neighbors in spa_idx.
    No reciprocal-NN check (queries are external, not in the indices).
    """
    shw_idx, shw_meta = _load_index_and_meta(
        filtered_dir / "faiss_shw.index",
        filtered_dir / "faiss_shw_meta.parquet",
    )
    spa_idx, spa_meta = _load_index_and_meta(
        filtered_dir / "faiss_spa.index",
        filtered_dir / "faiss_spa_meta.parquet",
    )

    rows: list[dict] = []
    for direction, queries, target_idx, target_meta, target_col in (
        ("spa_query", extra_spa_text, shw_idx, shw_meta, "shiwilu"),
        ("shw_query", extra_shw_text, spa_idx, spa_meta, "spanish"),
    ):
        if not queries:
            continue
        emb = encode(
            model, list(queries),
            batch_size=embedding_cfg.batch_size, use_e5=embedding_cfg.use_e5_prefixes, role="query"
        )
        sims, idxs = target_idx.search(emb, cfg.top_k)
        for q, q_text in enumerate(queries):
            for rank, (n_idx, n_sim) in enumerate(zip(idxs[q], sims[q])):
                n_idx = int(n_idx)
                if n_idx < 0 or float(n_sim) < cfg.min_ip:
                    continue
                if direction == "spa_query":
                    rows.append(
                        {
                            "spanish": q_text,
                            "shiwilu": target_meta[target_col].iloc[n_idx],
                            "ip": float(n_sim),
                            "rank": int(rank),
                            "source": "external_spa_query",
                        }
                    )
                else:
                    rows.append(
                        {
                            "shiwilu": q_text,
                            "spanish": target_meta[target_col].iloc[n_idx],
                            "ip": float(n_sim),
                            "rank": int(rank),
                            "source": "external_shw_query",
                        }
                    )
    df = pd.DataFrame(rows)
    info = {
        "mode": "external",
        "min_ip": cfg.min_ip,
        "top_k": cfg.top_k,
        "spa_queries": len(extra_spa_text),
        "shw_queries": len(extra_shw_text),
        "raw_candidates": int(len(df)),
    }
    return df, info


def to_canonical_dataframe(mined: pd.DataFrame) -> pd.DataFrame:
    """Convert mined (spanish, shiwilu, ip) candidates into the canonical
    bidirectional schema with origin_source='mined_v3_sbert'.
    """
    rows: list[dict] = []
    for i, row in mined.iterrows():
        pair_id = f"MINE{i:06d}"
        group_id = f"GMINE{i:06d}"
        score = float(row["ip"])
        for src_lang, tgt_lang, src, tgt in (
            ("shw", "spa", row["shiwilu"], row["spanish"]),
            ("spa", "shw", row["spanish"], row["shiwilu"]),
        ):
            rows.append(
                {
                    "id": f"{pair_id}__{src_lang}2{tgt_lang}",
                    "pair_id": pair_id,
                    "group_id": group_id,
                    "source": src,
                    "target": tgt,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "split": "train",
                    "has_audit_flags": False,
                    "origin_source": "mined_v3_sbert",
                    "score": score,
                    "label": "accepted",
                }
            )
    return pd.DataFrame(rows)
