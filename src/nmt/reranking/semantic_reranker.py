"""Phase 6: semantic reranker.

For each source, given an n-best list with NLLB sequence_scores:
    1. Convert sequence_scores -> softmax over the n candidates -> trans_prob.
    2. Encode source and each candidate with the frozen v3 SBERT.
    3. Compute cos_sim between source and each candidate.
    4. final_score = alpha * trans_prob + (1 - alpha) * cos_sim.
    5. Re-pick best candidate by final_score.

Run with the prescribed alpha=0.7 (translation weight) plus an ablation
sweep over {0.0, 0.3, 0.5, 0.7, 1.0} so the chosen 0.7/0.3 split is
empirically supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from ..preprocessing.semantic_filter import (
    SemanticFilterConfig,
    encode,
    load_embedding_model,
    resolve_device,
)


@dataclass(frozen=True)
class RerankerConfig:
    weight_translation: float
    weight_semantic: float
    prob_normalization: str
    embedding_path: Path
    use_e5_prefixes: bool
    batch_size: int
    device: str
    ablation_alphas: tuple[float, ...]

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path) -> "RerankerConfig":
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        emb = cfg["embedding_model"]
        return cls(
            weight_translation=float(cfg["weights"]["translation"]),
            weight_semantic=float(cfg["weights"]["semantic"]),
            prob_normalization=str(cfg.get("prob_normalization", "softmax_over_nbest")),
            embedding_path=project_root / emb["path"],
            use_e5_prefixes=bool(emb.get("use_e5_prefixes", False)),
            batch_size=int(emb.get("batch_size", 32)),
            device=str(emb.get("device", "auto")),
            ablation_alphas=tuple(cfg.get("ablation", {}).get("alphas", [0.0, 0.3, 0.5, 0.7, 1.0])),
        )

    def to_filter_config(self) -> SemanticFilterConfig:
        from ..preprocessing.semantic_filter import FilterThresholds

        return SemanticFilterConfig(
            model_path=self.embedding_path,
            use_e5_prefixes=self.use_e5_prefixes,
            batch_size=self.batch_size,
            device=self.device,
            fp16=False,
            thresholds=FilterThresholds(remove_below=0.45, flag_upper=0.60),
        )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _normalize_prob(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "softmax_over_nbest":
        return _softmax(scores, axis=-1)
    if mode == "exp_minmax":
        ex = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return ex / ex.sum(axis=-1, keepdims=True)
    raise ValueError(f"unknown prob_normalization mode: {mode!r}")


def _semantic_scores(
    model: SentenceTransformer,
    sources: list[str],
    candidate_lists: list[list[str]],
    *,
    batch_size: int,
    use_e5: bool,
) -> np.ndarray:
    """Return shape (n_sources, n_candidates) of cosine similarities."""
    n_src = len(sources)
    if n_src == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n_cand = len(candidate_lists[0])
    for cl in candidate_lists:
        if len(cl) != n_cand:
            raise ValueError("all n-best lists must have the same length")

    src_emb = encode(model, sources, batch_size=batch_size, use_e5=use_e5, role="passage")
    flat_cand = [c for cl in candidate_lists for c in cl]
    cand_emb = encode(model, flat_cand, batch_size=batch_size, use_e5=use_e5, role="passage")
    cand_emb = cand_emb.reshape(n_src, n_cand, -1)
    sims = (cand_emb * src_emb[:, None, :]).sum(axis=-1)
    return sims.astype(np.float32)


def rerank(
    predictions: list[dict],
    cfg: RerankerConfig,
    *,
    alpha: float | None = None,
) -> list[dict]:
    """Return a re-ranked copy of predictions. `alpha` overrides cfg.weight_translation."""
    if alpha is None:
        alpha = cfg.weight_translation
    beta = 1.0 - alpha

    if not predictions:
        return []

    sources = [p["source"] for p in predictions]
    candidate_lists = [
        [c["hypothesis"] for c in p["candidates"]] for p in predictions
    ]
    sequence_scores = np.asarray(
        [[c["sequence_score"] for c in p["candidates"]] for p in predictions],
        dtype=np.float64,
    )
    trans_probs = _normalize_prob(sequence_scores, cfg.prob_normalization).astype(np.float32)

    model = load_embedding_model(cfg.to_filter_config())
    sem_scores = _semantic_scores(
        model,
        sources,
        candidate_lists,
        batch_size=cfg.batch_size,
        use_e5=cfg.use_e5_prefixes,
    )

    final = alpha * trans_probs + beta * sem_scores
    best_idx = final.argmax(axis=-1)

    out: list[dict] = []
    for i, p in enumerate(predictions):
        candidates = []
        for j, cand in enumerate(p["candidates"]):
            candidates.append(
                {
                    **cand,
                    "trans_prob": float(trans_probs[i, j]),
                    "cos_sim": float(sem_scores[i, j]),
                    "final_score": float(final[i, j]),
                }
            )
        new_p = {**p, "candidates": candidates}
        chosen = candidates[int(best_idx[i])]
        new_p["hypothesis"] = chosen["hypothesis"]
        new_p["sequence_score"] = chosen["sequence_score"]
        new_p["chosen_rank"] = int(best_idx[i])
        new_p["alpha"] = alpha
        out.append(new_p)

    return out


def evaluate_rerank_ablation(
    predictions: list[dict],
    cfg: RerankerConfig,
    metrics_cfg,                 # MetricsConfig (avoid circular import)
    *,
    alphas: Iterable[float] | None = None,
) -> dict:
    """Sweep alpha values, computing chrF++ + BLEU per direction for each."""
    from ..evaluation.metrics import compute_bleu_chrf

    alphas = list(alphas) if alphas is not None else list(cfg.ablation_alphas)
    by_alpha: dict[str, dict] = {}

    for alpha in alphas:
        reranked = rerank(predictions, cfg, alpha=alpha)
        per_dir: dict[str, dict[str, float]] = {}
        for direction in {p["direction"] for p in reranked}:
            sub = [p for p in reranked if p["direction"] == direction]
            metrics = compute_bleu_chrf(
                [p["hypothesis"] for p in sub],
                [p["reference"] for p in sub],
                metrics_cfg,
            )
            per_dir[direction] = {
                "n": len(sub),
                "bleu": metrics["bleu"],
                "chrf_pp": metrics["chrf_pp"],
            }
        if {"shw2spa", "spa2shw"}.issubset(per_dir.keys()):
            avg_chrf = float(np.mean([per_dir[d]["chrf_pp"] for d in ("shw2spa", "spa2shw")]))
            avg_bleu = float(np.mean([per_dir[d]["bleu"] for d in ("shw2spa", "spa2shw")]))
        else:
            avg_chrf = float(np.mean([m["chrf_pp"] for m in per_dir.values()]))
            avg_bleu = float(np.mean([m["bleu"] for m in per_dir.values()]))
        by_alpha[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "directions": per_dir,
            "avg_chrf_pp": avg_chrf,
            "avg_bleu": avg_bleu,
        }
    best_alpha = max(by_alpha.values(), key=lambda v: v["avg_chrf_pp"])
    return {
        "alphas": list(alphas),
        "by_alpha": by_alpha,
        "best": {
            "alpha": best_alpha["alpha"],
            "avg_chrf_pp": best_alpha["avg_chrf_pp"],
            "avg_bleu": best_alpha["avg_bleu"],
        },
        "selection_metric": "avg_chrf_pp",
    }
