"""Phase 2b: FAISS IndexFlatIP per side (Shiwilu / Spanish).

Indices are built on the *accepted* train rows so downstream stages
(reranker hallucination checks, Phase 7 mining) only see clean neighbors.
Embeddings are L2-normalized so inner-product = cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .semantic_filter import encode


@dataclass(frozen=True)
class SideArtifacts:
    side: str               # "shiwilu" or "spanish"
    index_path: Path
    meta_path: Path
    n_vectors: int
    dim: int


def _accepted_pairs_unique_text(filtered_train_csv: Path) -> pd.DataFrame:
    """Return one row per pair_id with cleaned shiwilu / spanish columns.

    The filtered_train_csv only contains directional rows labeled `accepted`.
    """
    df = pd.read_csv(filtered_train_csv, encoding="utf-8-sig")
    shw = df[df["source_lang"] == "shw"][["pair_id", "source", "target", "origin_source", "score"]]
    shw = shw.rename(columns={"source": "shiwilu", "target": "spanish"})
    return shw.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)


def build_side_index(
    pairs_df: pd.DataFrame,
    side: str,
    model: SentenceTransformer,
    out_dir: Path,
    *,
    batch_size: int,
    use_e5: bool,
) -> SideArtifacts:
    if side == "shiwilu":
        text_col = "shiwilu"
    elif side == "spanish":
        text_col = "spanish"
    else:
        raise ValueError(f"unknown side {side!r}; expected 'shiwilu' or 'spanish'")

    texts = pairs_df[text_col].tolist()
    embeddings = encode(model, texts, batch_size=batch_size, use_e5=use_e5, role="passage")
    if embeddings.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape: {embeddings.shape}")
    dim = int(embeddings.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings))

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / f"faiss_{ {'shiwilu': 'shw', 'spanish': 'spa'}[side] }.index"
    meta_path = out_dir / f"faiss_{ {'shiwilu': 'shw', 'spanish': 'spa'}[side] }_meta.parquet"

    faiss.write_index(index, str(index_path))

    meta = pairs_df[["pair_id", "shiwilu", "spanish", "origin_source", "score"]].copy()
    meta["row_idx"] = meta.index.astype("int64")
    meta["text"] = meta[text_col]
    meta = meta[["row_idx", "pair_id", "text", "shiwilu", "spanish", "origin_source", "score"]]
    meta.to_parquet(meta_path, index=False)

    return SideArtifacts(
        side=side,
        index_path=index_path,
        meta_path=meta_path,
        n_vectors=len(texts),
        dim=dim,
    )


def near_duplicate_check(
    artifact: SideArtifacts,
    pairs_df: pd.DataFrame,
    *,
    ip_threshold: float = 0.98,
) -> dict:
    """Self-similarity scan to flag near-duplicates (no auto-removal)."""
    index = faiss.read_index(str(artifact.index_path))
    text_col = {"shiwilu": "shiwilu", "spanish": "spanish"}[artifact.side]
    texts = pairs_df[text_col].tolist()

    # Re-encode the same vectors (deterministic) for the scan.
    # Use ntotal == len(texts), so index.reconstruct_n works.
    n = index.ntotal
    vecs = np.vstack([index.reconstruct(i) for i in range(n)]).astype(np.float32)

    sims, idxs = index.search(vecs, 2)  # nearest is self, then real nearest neighbor
    flagged: list[dict] = []
    for i in range(n):
        # idxs[i, 0] is self with sim ~ 1.0
        nbr = int(idxs[i, 1])
        sim = float(sims[i, 1])
        if nbr < 0:
            continue
        if sim >= ip_threshold and nbr != i:
            flagged.append(
                {
                    "row_a": int(i),
                    "row_b": nbr,
                    "pair_id_a": pairs_df["pair_id"].iloc[i],
                    "pair_id_b": pairs_df["pair_id"].iloc[nbr],
                    "ip": sim,
                    "text_a": texts[i],
                    "text_b": texts[nbr],
                }
            )
    return {
        "side": artifact.side,
        "ip_threshold": ip_threshold,
        "flagged_pairs": flagged,
        "flagged_count": len(flagged),
        "total_vectors": n,
    }


def collect_pairs_for_indexing(filtered_dir: Path) -> pd.DataFrame:
    return _accepted_pairs_unique_text(filtered_dir / "train.csv")
