"""Phase 2a: semantic filtering with the frozen v3 SBERT.

The filter is computed once per pair_id (not per direction): both directional
rows for a given pair_id share the same score and label.

Thresholds (config/nmt/filter.yaml):
    score < remove_below              -> "removed"
    remove_below <= score <= flag_upper -> "flagged_for_review"
    score > flag_upper                -> "accepted"

The filter is applied only to train. Valid/test pass through with the score
column attached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class FilterThresholds:
    remove_below: float
    flag_upper: float


@dataclass(frozen=True)
class SemanticFilterConfig:
    model_path: Path
    use_e5_prefixes: bool
    batch_size: int
    device: str
    fp16: bool
    thresholds: FilterThresholds

    @classmethod
    def from_yaml(cls, yaml_path: Path, project_root: Path) -> "SemanticFilterConfig":
        with yaml_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        emb = cfg["embedding_model"]
        thr = cfg["thresholds"]
        return cls(
            model_path=project_root / emb["path"],
            use_e5_prefixes=bool(emb.get("use_e5_prefixes", False)),
            batch_size=int(emb.get("batch_size", 32)),
            device=str(emb.get("device", "auto")),
            fp16=bool(emb.get("fp16", False)),
            thresholds=FilterThresholds(
                remove_below=float(thr["remove_below"]),
                flag_upper=float(thr["flag_upper"]),
            ),
        )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedding_model(cfg: SemanticFilterConfig) -> SentenceTransformer:
    device = resolve_device(cfg.device)
    model = SentenceTransformer(str(cfg.model_path), device=device)
    model.eval()
    return model


def _maybe_prefix(texts: Iterable[str], use_e5: bool, role: str) -> list[str]:
    if not use_e5:
        return list(texts)
    tag = "query: " if role == "query" else "passage: "
    return [f"{tag}{t}" for t in texts]


def encode(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    use_e5: bool = False,
    role: str = "passage",
) -> np.ndarray:
    """Encode + L2-normalize. Cosine similarity = dot product."""
    prepared = _maybe_prefix(texts, use_e5=use_e5, role=role)
    with torch.no_grad():
        emb = model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return np.asarray(emb, dtype=np.float32)


def compute_pair_scores(
    model: SentenceTransformer,
    shiwilu_texts: list[str],
    spanish_texts: list[str],
    *,
    batch_size: int,
    use_e5: bool = False,
) -> np.ndarray:
    if len(shiwilu_texts) != len(spanish_texts):
        raise ValueError("shiwilu and spanish texts must have equal length")
    e_shw = encode(model, shiwilu_texts, batch_size=batch_size, use_e5=use_e5, role="passage")
    e_spa = encode(model, spanish_texts, batch_size=batch_size, use_e5=use_e5, role="passage")
    scores = (e_shw * e_spa).sum(axis=1)
    return scores.astype(np.float32)


def label_score(score: float, thresholds: FilterThresholds) -> str:
    if score < thresholds.remove_below:
        return "removed"
    if score <= thresholds.flag_upper:
        return "flagged_for_review"
    return "accepted"


def _unique_pairs(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """One row per pair_id with explicit shiwilu/spanish columns."""
    shw_rows = canonical_df[canonical_df["source_lang"] == "shw"][["pair_id", "source", "target"]]
    shw_rows = shw_rows.rename(columns={"source": "shiwilu", "target": "spanish"})
    return shw_rows.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)


def score_split(
    canonical_csv: Path,
    model: SentenceTransformer,
    cfg: SemanticFilterConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (canonical_with_score, per_pair_score_table)."""
    df = pd.read_csv(canonical_csv, encoding="utf-8-sig")
    pairs = _unique_pairs(df)
    scores = compute_pair_scores(
        model,
        pairs["shiwilu"].tolist(),
        pairs["spanish"].tolist(),
        batch_size=cfg.batch_size,
        use_e5=cfg.use_e5_prefixes,
    )
    pairs["score"] = scores
    pairs["label"] = [label_score(float(s), cfg.thresholds) for s in scores]
    score_map = pairs.set_index("pair_id")[["score", "label"]]
    df = df.join(score_map, on="pair_id")
    return df, pairs


def histogram(scores: np.ndarray, bins: list[float]) -> dict[str, int]:
    counts, _ = np.histogram(scores, bins=bins)
    out: dict[str, int] = {}
    for i, c in enumerate(counts):
        lo, hi = bins[i], bins[i + 1]
        out[f"[{lo:.2f},{hi:.2f})"] = int(c)
    return out


def per_origin_stats(pairs_df: pd.DataFrame, canonical_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    origin_lookup = (
        canonical_df[["pair_id", "origin_source"]].drop_duplicates(subset=["pair_id"]).set_index("pair_id")
    )
    enriched = pairs_df.join(origin_lookup, on="pair_id")
    out: dict[str, dict[str, float]] = {}
    for origin, group in enriched.groupby("origin_source"):
        out[str(origin)] = {
            "count": int(len(group)),
            "mean": float(group["score"].mean()),
            "std": float(group["score"].std(ddof=0)) if len(group) > 0 else 0.0,
            "min": float(group["score"].min()) if len(group) > 0 else 0.0,
            "max": float(group["score"].max()) if len(group) > 0 else 0.0,
        }
    return out


def write_partition(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
