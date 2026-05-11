"""
Controlled hard negative mining and training for Shiwlu-Spanish embeddings.

This experiment keeps v1 intact. It first mines semi-hard negatives with v1,
validates their margins and metadata, and only then trains a separate model:
models/sentence_transformers/finetuned_v2_hn_controlled.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from evaluate_retrieval import evaluate_model, load_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
MODEL_DIR = PROJECT_ROOT / "models" / "sentence_transformers"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"
CONTROLLED_HN_REPORTS_DIR = REPORTS_DIR / "controlled_hn"

V1_DIR = MODEL_DIR / "finetuned_v1"
V2_HN_DIR = MODEL_DIR / "finetuned_v2_hn_controlled"
V2_HN_HARD_DIR = MODEL_DIR / "finetuned_v2_hn_controlled_hard"

BASE_MODEL = "intfloat/multilingual-e5-small"
STAGES = ("mine", "train", "evaluate", "all")
VARIANTS = ("all", "hard", "medium")

TOP_K = 50
HARD_MIN_MARGIN = 0.05
HARD_MAX_MARGIN = 0.15
MEDIUM_MAX_MARGIN = 0.30
DEFAULT_SAMPLE_SIZE = 100


@dataclass(frozen=True)
class MiningStats:
    total_anchors: int
    anchors_with_negative: int
    anchors_without_negative: int
    same_group_discards: int
    duplicate_discards: int
    too_hard_discards: int
    too_easy_discards: int

    @property
    def valid_anchor_ratio(self) -> float:
        return self.anchors_with_negative / max(self.total_anchors, 1)


class HardNegativeDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, *, bidirectional: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.bidirectional = bidirectional

    def __len__(self) -> int:
        multiplier = 2 if self.bidirectional else 1
        return len(self.rows) * multiplier

    def __getitem__(self, idx: int) -> dict[str, str]:
        if self.bidirectional:
            row = self.rows.iloc[idx // 2]
            reverse = idx % 2 == 1
        else:
            row = self.rows.iloc[idx]
            reverse = False

        if reverse:
            if "negative_anchor" not in row:
                raise KeyError("Bidirectional hard-negative training requires a negative_anchor column.")
            return {
                "anchor": f"query: {str(row['positive']).strip()}",
                "positive": f"passage: {str(row['anchor']).strip()}",
                "negative": f"passage: {str(row['negative_anchor']).strip()}",
            }

        return {
            "anchor": f"query: {str(row['anchor']).strip()}",
            "positive": f"passage: {str(row['positive']).strip()}",
            "negative": f"passage: {str(row['negative']).strip()}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled hard negative experiment.")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--model", default=str(V1_DIR), help="Modelo usado para minería/evaluación.")
    parser.add_argument("--base-model", default=str(V1_DIR), help="Modelo inicial para entrenamiento.")
    parser.add_argument("--experiment-name", default="v2_hn_controlled")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--variant", choices=VARIANTS, default="all")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Entrena también la dirección Shiwlu->español usando negativos españoles emparejados."
    )
    parser.add_argument(
        "--splits-variant",
        choices=("main", "xl"),
        default="main",
        help="Usa data/processed/04_splits (main) o 04_splits_xl (xl).",
    )
    return parser.parse_args()


def controlled_model_dir(experiment_name: str, variant: str) -> Path:
    if experiment_name == "v2_hn_controlled" and variant == "hard":
        return V2_HN_HARD_DIR
    if experiment_name == "v2_hn_controlled" and variant == "all":
        return V2_HN_DIR
    if variant == "medium":
        return MODEL_DIR / "finetuned_v2_hn_controlled_medium"
    if variant == "all":
        return MODEL_DIR / experiment_name
    return MODEL_DIR / f"{experiment_name}_{variant}"


def controlled_tag(experiment_name: str, variant: str) -> str:
    if experiment_name == "v2_hn_controlled" and variant == "hard":
        return "v2_hn_controlled_hard"
    if variant == "medium":
        return f"{experiment_name}_medium"
    if variant == "all":
        return experiment_name
    return f"{experiment_name}_{variant}"


def controlled_report_dir(experiment_name: str) -> Path:
    if experiment_name == "v2_hn_controlled":
        return CONTROLLED_HN_REPORTS_DIR
    return CONTROLLED_HN_REPORTS_DIR / experiment_name


def negatives_path(experiment_name: str) -> Path:
    if experiment_name == "v2_hn_controlled":
        return SPLITS_DIR / "train_hard_negatives_controlled.csv"
    return SPLITS_DIR / f"train_hard_negatives_{experiment_name}.csv"


def negatives_sample_path(experiment_name: str) -> Path:
    if experiment_name == "v2_hn_controlled":
        return controlled_report_dir(experiment_name) / "hard_negatives_controlled_sample.csv"
    return controlled_report_dir(experiment_name) / f"hard_negatives_{experiment_name}_sample.csv"


def negatives_report_path(experiment_name: str) -> Path:
    if experiment_name == "v2_hn_controlled":
        return controlled_report_dir(experiment_name) / "hard_negatives_controlled_report.json"
    return controlled_report_dir(experiment_name) / f"hard_negatives_{experiment_name}_report.json"


def negatives_path_for_args(args: argparse.Namespace) -> Path:
    base = negatives_path(args.experiment_name)
    if args.splits_variant == "xl":
        return base.with_name(base.stem + "_xl.csv")
    return base


def resolved_base_model(args: argparse.Namespace) -> str:
    if args.base_model != str(V1_DIR):
        return args.base_model
    if "e5_base" in args.experiment_name:
        return str(MODEL_DIR / "v1_e5_base")
    return args.base_model


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, a_min=1e-12, a_max=None)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "max": float(arr.max()),
    }


def difficulty_from_margin(margin: float) -> str | None:
    if HARD_MIN_MARGIN <= margin <= HARD_MAX_MARGIN:
        return "hard"
    if HARD_MAX_MARGIN < margin <= MEDIUM_MAX_MARGIN:
        return "medium"
    return None


def encode_train_split(model: SentenceTransformer, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    anchors = [f"query: {text.strip()}" for text in train_df["ESP_normalizado"].astype(str)]
    passages = [f"passage: {text.strip()}" for text in train_df["SHIWILU_normalizado"].astype(str)]
    anchor_embs = model.encode(anchors, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    passage_embs = model.encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    return l2_normalize(anchor_embs), l2_normalize(passage_embs)


def mine_negatives(args: argparse.Namespace) -> pd.DataFrame:
    start_time = datetime.now(timezone.utc)
    random.seed(args.seed)
    output_report_dir = controlled_report_dir(args.experiment_name)
    output_report_dir.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train", splits_variant=args.splits_variant)
    model = SentenceTransformer(args.model)
    anchor_embs, passage_embs = encode_train_split(model, train_df)
    sim_matrix = anchor_embs @ passage_embs.T

    rows = []
    same_group_discards = 0
    duplicate_discards = 0
    too_hard_discards = 0
    too_easy_discards = 0
    anchors_with_negative = 0

    group_ids = train_df["group_id"].astype(str).tolist()
    shiwilu_texts = train_df["SHIWILU_normalizado"].astype(str).tolist()

    for anchor_idx, row in train_df.reset_index(drop=True).iterrows():
        sim_positive = float(sim_matrix[anchor_idx, anchor_idx])
        candidate_indices = np.argsort(-sim_matrix[anchor_idx])[: args.top_k + 1]
        selected_by_difficulty: dict[str, dict[str, Any]] = {}

        for candidate_idx in candidate_indices:
            candidate_idx = int(candidate_idx)
            if candidate_idx == anchor_idx:
                continue

            sim_negative = float(sim_matrix[anchor_idx, candidate_idx])
            margin = sim_positive - sim_negative
            difficulty = difficulty_from_margin(margin)

            if group_ids[candidate_idx] == group_ids[anchor_idx]:
                same_group_discards += 1
                continue
            if shiwilu_texts[candidate_idx] == shiwilu_texts[anchor_idx]:
                duplicate_discards += 1
                continue
            if sim_negative >= sim_positive or margin < HARD_MIN_MARGIN:
                too_hard_discards += 1
                continue
            if difficulty is None:
                too_easy_discards += 1
                continue
            if difficulty in selected_by_difficulty:
                continue

            candidate = train_df.iloc[candidate_idx]
            selected_by_difficulty[difficulty] = {
                "pair_id": row["pair_id"],
                "group_id": row["group_id"],
                "anchor": row["ESP_normalizado"],
                "positive": row["SHIWILU_normalizado"],
                "negative": candidate["SHIWILU_normalizado"],
                "negative_anchor": candidate["ESP_normalizado"],
                "negative_pair_id": candidate["pair_id"],
                "negative_group_id": candidate["group_id"],
                "difficulty": difficulty,
                "sim_positive": sim_positive,
                "sim_negative": sim_negative,
                "margin": margin,
                "negative_rank": int(np.where(candidate_indices == candidate_idx)[0][0] + 1),
            }

        if selected_by_difficulty:
            anchors_with_negative += 1
            if "hard" in selected_by_difficulty:
                rows.append(selected_by_difficulty["hard"])
            if "medium" in selected_by_difficulty:
                rows.append(selected_by_difficulty["medium"])

    output_negatives_path = negatives_path_for_args(args)
    output_sample_path = negatives_sample_path(args.experiment_name)
    output_report_path = negatives_report_path(args.experiment_name)

    mined_df = pd.DataFrame(rows)
    mined_df.to_csv(output_negatives_path, index=False, encoding="utf-8-sig")

    sample_df = mined_df.sample(
        n=min(args.sample_size, len(mined_df)),
        random_state=args.seed,
    ) if len(mined_df) else mined_df
    sample_df.to_csv(output_sample_path, index=False, encoding="utf-8-sig")

    stats = MiningStats(
        total_anchors=len(train_df),
        anchors_with_negative=anchors_with_negative,
        anchors_without_negative=len(train_df) - anchors_with_negative,
        same_group_discards=same_group_discards,
        duplicate_discards=duplicate_discards,
        too_hard_discards=too_hard_discards,
        too_easy_discards=too_easy_discards,
    )
    report = build_mining_report(mined_df, stats, args, start_time)
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Negativos minados: {output_negatives_path}")
    print(f"Muestra cualitativa: {output_sample_path}")
    print(f"Reporte: {output_report_path}")
    print(f"Auto-pass: {report['quality_gate']['auto_pass']}")
    return mined_df


def build_mining_report(
    mined_df: pd.DataFrame,
    stats: MiningStats,
    args: argparse.Namespace,
    start_time: datetime,
) -> dict[str, Any]:
    elapsed = datetime.now(timezone.utc) - start_time
    margins = mined_df["margin"].astype(float).tolist() if len(mined_df) else []
    sim_positive = mined_df["sim_positive"].astype(float).tolist() if len(mined_df) else []
    sim_negative = mined_df["sim_negative"].astype(float).tolist() if len(mined_df) else []
    difficulty_counts = mined_df["difficulty"].value_counts().to_dict() if len(mined_df) else {}

    auto_pass = (
        stats.valid_anchor_ratio >= 0.5
        and len(mined_df) > 0
        and all(margin >= HARD_MIN_MARGIN for margin in margins)
    )

    return {
        "pipeline": "hard_negative_controlled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_model": args.model,
        "top_k": args.top_k,
        "margin_rules": {
            "hard": [HARD_MIN_MARGIN, HARD_MAX_MARGIN],
            "medium": [HARD_MAX_MARGIN, MEDIUM_MAX_MARGIN],
        },
        "counts": {
            "total_anchors": stats.total_anchors,
            "anchors_with_negative": stats.anchors_with_negative,
            "anchors_without_negative": stats.anchors_without_negative,
            "valid_anchor_ratio": stats.valid_anchor_ratio,
            "total_negative_rows": int(len(mined_df)),
            "difficulty_counts": difficulty_counts,
            "same_group_discards": stats.same_group_discards,
            "duplicate_discards": stats.duplicate_discards,
            "too_hard_discards": stats.too_hard_discards,
            "too_easy_discards": stats.too_easy_discards,
        },
        "similarity": {
            "sim_positive": summarize(sim_positive),
            "sim_negative": summarize(sim_negative),
            "margin": summarize(margins),
        },
        "quality_gate": {
            "auto_pass": auto_pass,
            "reason": (
                "Valid anchor ratio and margins are acceptable."
                if auto_pass
                else "Do not train automatically; inspect mined negatives first."
            ),
        },
        "artifacts": {
            "negatives_csv": str(negatives_path_for_args(args)),
            "sample_csv": str(negatives_sample_path(args.experiment_name)),
            "report_json": str(negatives_report_path(args.experiment_name)),
        },
        "elapsed_seconds": elapsed.total_seconds(),
    }


def tensor_to_device(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


def collate_triplets(batch: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "anchor": [item["anchor"] for item in batch],
        "positive": [item["positive"] for item in batch],
        "negative": [item["negative"] for item in batch],
    }


def train_controlled(args: argparse.Namespace) -> None:
    input_report_path = negatives_report_path(args.experiment_name)
    input_negatives_path = negatives_path_for_args(args)
    if not input_report_path.exists() or not input_negatives_path.exists():
        raise FileNotFoundError("Run --stage mine before training.")

    report = json.loads(input_report_path.read_text(encoding="utf-8"))
    if not report["quality_gate"]["auto_pass"]:
        raise RuntimeError("Mining quality gate did not pass; inspect negatives before training.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    mined_df = pd.read_csv(input_negatives_path, encoding="utf-8-sig")
    if args.variant != "all":
        mined_df = mined_df[mined_df["difficulty"] == args.variant].copy()
    if mined_df.empty:
        raise RuntimeError(f"No negatives available for variant={args.variant}.")

    dataset = HardNegativeDataset(mined_df, bidirectional=args.bidirectional)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_triplets)

    base_model = resolved_base_model(args)
    model = SentenceTransformer(base_model)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            anchor_features = tensor_to_device(model.preprocess(batch["anchor"]), device)
            positive_features = tensor_to_device(model.preprocess(batch["positive"]), device)
            negative_features = tensor_to_device(model.preprocess(batch["negative"]), device)

            anchor_emb = model(anchor_features)["sentence_embedding"]
            positive_emb = model(positive_features)["sentence_embedding"]
            negative_emb = model(negative_features)["sentence_embedding"]

            anchor_emb = torch.nn.functional.normalize(anchor_emb, dim=-1)
            positive_emb = torch.nn.functional.normalize(positive_emb, dim=-1)
            negative_emb = torch.nn.functional.normalize(negative_emb, dim=-1)

            candidates = torch.cat([positive_emb, negative_emb], dim=0)
            logits = anchor_emb @ candidates.T / args.temperature
            labels = torch.arange(anchor_emb.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f}")

    output_dir = controlled_model_dir(args.experiment_name, args.variant)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    print(f"Modelo guardado en {output_dir}")


def evaluate_controlled(args: argparse.Namespace) -> dict[str, Any]:
    experiment_name = args.experiment_name
    variant = args.variant
    output_dir = controlled_model_dir(experiment_name, variant)
    tag = controlled_tag(experiment_name, variant)
    model = SentenceTransformer(str(output_dir))
    test_df = load_split("test", splits_variant=args.splits_variant)
    start_time = datetime.now(timezone.utc)
    metrics = evaluate_model(
        model,
        test_df,
        tag,
        str(output_dir),
        start_time,
    )

    training_report = {
        "pipeline": f"finetune_st_{tag}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(output_dir),
        "base_model": resolved_base_model(args),
        "experiment_name": experiment_name,
        "variant": variant,
        "loss": "explicit_negative_mnrl_style_cross_entropy",
        "bidirectional": args.bidirectional,
        "retrieval_metrics": metrics,
        "acceptance_reference": {
            "v1_recall@1": 0.5109034267912772,
            "v1_recall@5": 0.778816199376947,
            "v1_recall@10": 0.8691588785046729,
            "v1_mrr": 0.632462097901643,
        },
    }
    report_dir = REPORTS_DIR / tag
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{tag}_training.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(training_report, f, ensure_ascii=False, indent=2)
    print(f"Reporte entrenamiento: {report_path}")
    return metrics


def main() -> None:
    args = parse_args()
    if args.stage in {"mine", "all"}:
        mine_negatives(args)
    if args.stage in {"train", "all"}:
        train_controlled(args)
    if args.stage in {"evaluate", "all"}:
        evaluate_controlled(args)


if __name__ == "__main__":
    main()
