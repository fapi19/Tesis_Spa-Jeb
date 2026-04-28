"""
Post-training analysis utilities for Shiwlu-Spanish embedding experiments.

Generates:
- R@1 error analysis for a retrieval model.
- Hard negative validation report.
- Frozen candidate metadata for the current best model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from evaluate_retrieval import load_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"
REPORTS_PREPROCESSING_DIR = REPORTS_DIR / "preprocessing"
CONTROLLED_HN_REPORTS_DIR = REPORTS_DIR / "controlled_hn"
MODEL_DIR = PROJECT_ROOT / "models" / "sentence_transformers"

DEFAULT_MODEL = MODEL_DIR / "finetuned_v2_hn_controlled"
DEFAULT_TAG = "v2_hn_controlled"
NEGATIVES_PATH = SPLITS_DIR / "train_hard_negatives_controlled.csv"


def report_dir_for_tag(tag: str) -> Path:
    if tag in {"v2_hn_controlled", "v2_hn_controlled_hard"}:
        return REPORTS_DIR / tag
    if tag == "v1":
        return REPORTS_DIR / "v1"
    if tag == "baseline":
        return REPORTS_DIR / "baseline"
    return REPORTS_DIR / "experiments" / tag


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, a_min=1e-12, a_max=None)


def build_positive_indices(df: pd.DataFrame) -> list[set[int]]:
    group_ids = df["group_id"].astype(str).tolist()
    group_to_indices: dict[str, set[int]] = {}
    for idx, group_id in enumerate(group_ids):
        group_to_indices.setdefault(group_id, set()).add(idx)
    return [group_to_indices[group_id] for group_id in group_ids]


def first_positive_rank(sorted_indices: np.ndarray, positives: set[int]) -> int:
    for rank, idx in enumerate(sorted_indices, start=1):
        if int(idx) in positives:
            return rank
    raise ValueError("No positive candidate found.")


def shiwilu_token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def heuristic_error_type(row: pd.Series, top1: pd.Series, rank: int, score_gap: float) -> str:
    if bool(row.get("has_audit_flags", False)):
        return "gold_has_audit_flag"
    if len(str(row["SHIWILU_normalizado"])) > 120 or len(str(row["ESP_normalizado"])) > 160:
        return "long_or_narrative_case"
    if score_gap < 0.05:
        return "close_score_ambiguity"
    if shiwilu_token_overlap(str(row["SHIWILU_normalizado"]), str(top1["SHIWILU_normalizado"])) > 0:
        return "shared_shiwilu_tokens"
    if rank > 50:
        return "severe_retrieval_failure"
    return "semantic_confusion"


def encode_split(model: SentenceTransformer, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    queries = [f"query: {text.strip()}" for text in df["ESP_normalizado"].astype(str)]
    passages = [f"passage: {text.strip()}" for text in df["SHIWILU_normalizado"].astype(str)]
    query_embs = model.encode(queries, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    passage_embs = model.encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    return l2_normalize(query_embs), l2_normalize(passage_embs)


def run_error_analysis(model_path: Path, tag: str, sample_size: int) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_split("test")
    model = SentenceTransformer(str(model_path))
    query_embs, passage_embs = encode_split(model, df)
    sim_matrix = query_embs @ passage_embs.T
    positive_indices = build_positive_indices(df)

    error_rows = []
    all_rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        scores = sim_matrix[idx]
        sorted_indices = np.argsort(-scores)
        positives = positive_indices[idx]
        rank = first_positive_rank(sorted_indices, positives)
        top1_idx = int(sorted_indices[0])
        top1 = df.iloc[top1_idx]
        best_positive_score = float(max(scores[list(positives)]))
        top1_score = float(scores[top1_idx])
        score_gap = top1_score - best_positive_score
        top1_is_positive = top1_idx in positives
        output_row = {
            "pair_id": row["pair_id"],
            "group_id": row["group_id"],
            "rank": rank,
            "top1_is_positive": top1_is_positive,
            "positive_count": len(positives),
            "esp": row["ESP_normalizado"],
            "shiwilu_correct": row["SHIWILU_normalizado"],
            "top1_pair_id": top1["pair_id"],
            "top1_group_id": top1["group_id"],
            "shiwilu_top1": top1["SHIWILU_normalizado"],
            "correct_score": best_positive_score,
            "top1_score": top1_score,
            "score_gap": score_gap,
            "heuristic_error_type": "correct_top1",
        }
        if not top1_is_positive:
            output_row["heuristic_error_type"] = heuristic_error_type(row, top1, rank, score_gap)
            error_rows.append(output_row)
        all_rows.append(output_row)

    output_dir = report_dir_for_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{tag}_r1_error_analysis_full.csv"
    sample_path = output_dir / f"{tag}_r1_error_analysis_review_sample.csv"
    summary_path = output_dir / f"{tag}_r1_error_analysis_summary.json"

    error_df = pd.DataFrame(error_rows).sort_values(["rank", "score_gap"], ascending=[False, False])
    all_df = pd.DataFrame(all_rows)
    error_df.to_csv(full_path, index=False, encoding="utf-8-sig")
    error_df.head(sample_size).to_csv(sample_path, index=False, encoding="utf-8-sig")

    summary = {
        "pipeline": "r1_error_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "tag": tag,
        "total_examples": int(len(df)),
        "top1_errors": int(len(error_df)),
        "top1_accuracy": float(1.0 - len(error_df) / max(len(df), 1)),
        "heuristic_error_counts": dict(Counter(error_df["heuristic_error_type"])) if len(error_df) else {},
        "artifacts": {
            "full_csv": str(full_path),
            "review_sample_csv": str(sample_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def validate_negatives(sample_size: int) -> dict[str, Any]:
    CONTROLLED_HN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    negatives = pd.read_csv(NEGATIVES_PATH, encoding="utf-8-sig")
    risk_rows = []
    for _, row in negatives.iterrows():
        risks = []
        if str(row["group_id"]) == str(row["negative_group_id"]):
            risks.append("same_group")
        if str(row["positive"]) == str(row["negative"]):
            risks.append("exact_duplicate")
        if float(row["margin"]) < 0.05:
            risks.append("below_margin")
        if float(row["margin"]) < 0.06:
            risks.append("very_close_margin")
        if shiwilu_token_overlap(str(row["positive"]), str(row["negative"])) > 0:
            risks.append("shared_shiwilu_tokens")
        if risks:
            risk_row = row.to_dict()
            risk_row["risk_flags"] = "|".join(risks)
            risk_rows.append(risk_row)

    risk_df = pd.DataFrame(risk_rows)
    risk_path = CONTROLLED_HN_REPORTS_DIR / "hard_negatives_controlled_risk_review.csv"
    summary_path = CONTROLLED_HN_REPORTS_DIR / "hard_negatives_controlled_validation.json"
    if len(risk_df):
        risk_df.head(sample_size).to_csv(risk_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(risk_path, index=False, encoding="utf-8-sig")

    summary = {
        "pipeline": "hard_negative_validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_negatives": int(len(negatives)),
        "risk_rows": int(len(risk_df)),
        "risk_ratio": float(len(risk_df) / max(len(negatives), 1)),
        "risk_counts": dict(Counter(flag for row in risk_rows for flag in row["risk_flags"].split("|"))),
        "artifacts": {
            "risk_review_csv": str(risk_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def freeze_candidate(model_path: Path, tag: str) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = report_dir_for_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS_PREPROCESSING_DIR / "preprocess_manifest.json"
    mining_report_path = CONTROLLED_HN_REPORTS_DIR / "hard_negatives_controlled_report.json"
    retrieval_path = output_dir / f"{tag}_retrieval.json"
    training_path = output_dir / f"{tag}_training.json"

    metadata = {
        "pipeline": "freeze_embedding_candidate",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_model": tag,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "data": {
            "preprocess_manifest": str(manifest_path),
            "preprocess_manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else "",
            "train_csv": str(SPLITS_DIR / "train.csv"),
            "valid_csv": str(SPLITS_DIR / "valid.csv"),
            "test_csv": str(SPLITS_DIR / "test.csv"),
        },
        "reports": {
            "mining_report": str(mining_report_path),
            "retrieval_report": str(retrieval_path),
            "training_report": str(training_path),
        },
        "decision": "candidate_final_embedding_model",
        "next_step": "Use this model as the embedding candidate unless later ablations beat it.",
    }

    output_path = output_dir / f"{tag}_freeze_metadata.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze embedding retrieval results and freeze metadata.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--skip-error-analysis", action="store_true")
    parser.add_argument("--skip-negative-validation", action="store_true")
    parser.add_argument("--skip-freeze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_error_analysis:
        summary = run_error_analysis(args.model, args.tag, args.sample_size)
        print(f"Error analysis: {summary['artifacts']['summary_json']}")
    if not args.skip_negative_validation:
        validation = validate_negatives(args.sample_size)
        print(f"Negative validation: {validation['artifacts']['summary_json']}")
    if not args.skip_freeze:
        metadata = freeze_candidate(args.model, args.tag)
        print(f"Freeze metadata: {report_dir_for_tag(args.tag) / f'{args.tag}_freeze_metadata.json'}")
        print(f"Candidate: {metadata['candidate_model']}")


if __name__ == "__main__":
    main()
