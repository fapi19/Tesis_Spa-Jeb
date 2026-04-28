"""
finetune_st.py
Fine-tuning de Sentence Transformers para embeddings cross-lingual español-shiwilu.

Pipeline por etapas: baseline → v1 (pares) → minería de negativos → v2 (triplets).

Uso:
    poetry run python src/embeddings/finetune_st.py --stage baseline
    poetry run python src/embeddings/finetune_st.py --stage v1 --epochs 10
    poetry run python src/embeddings/finetune_st.py --stage mine-negatives
    poetry run python src/embeddings/finetune_st.py --stage v2 --epochs 5

Etapas:
    baseline         Evalúa modelo pre-entrenado sin fine-tuning
    v1               Fine-tuning con pares (MultipleNegativesRankingLoss)
    mine-negatives   Minería de hard negatives desde modelo v1
    v2               Fine-tuning con triplets (TripletLoss) desde modelo v1

Salida:
    models/sentence_transformers/finetuned_v1/   - Modelo fine-tuned v1
    models/sentence_transformers/finetuned_v2/   - Modelo fine-tuned v2
    data/processed/04_splits/train_triplets.csv  - Triplets con hard negatives
    reports/04_embeddings/*_retrieval.json       - Métricas por etapa
    reports/04_embeddings/*_qualitative.csv      - Análisis cualitativo
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.training_args import BatchSamplers

from evaluate_retrieval import evaluate_model, load_split, report_dir_for_tag

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
MODEL_DIR = PROJECT_ROOT / "models" / "sentence_transformers"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"

BASE_MODEL = "intfloat/multilingual-e5-small"
V1_DIR = MODEL_DIR / "finetuned_v1"
V2_DIR = MODEL_DIR / "finetuned_v2"
TRIPLETS_PATH = SPLITS_DIR / "train_triplets.csv"

STAGES = ("baseline", "v1", "mine-negatives", "v2")


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning de Sentence Transformers cross-lingual español-shiwilu"
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=STAGES,
        help="Etapa del pipeline a ejecutar"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=BASE_MODEL,
        help=f"Modelo base para baseline/v1 (default: {BASE_MODEL})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de epochs (default: 10 para v1, 5 para v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size de entrenamiento (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    return parser.parse_args()


def build_pair_dataset(df: pd.DataFrame) -> Dataset:
    """Construye Dataset de pares con prefijos E5 asimétricos."""
    return Dataset.from_dict({
        "anchor": [f"query: {t.strip()}" for t in df["ESP_normalizado"].astype(str)],
        "positive": [f"passage: {t.strip()}" for t in df["SHIWILU_normalizado"].astype(str)],
    })


def build_triplet_dataset(triplets_df: pd.DataFrame) -> Dataset:
    """Construye Dataset de triplets con prefijos E5 asimétricos."""
    return Dataset.from_dict({
        "anchor": [f"query: {t.strip()}" for t in triplets_df["anchor"].astype(str)],
        "positive": [f"passage: {t.strip()}" for t in triplets_df["positive"].astype(str)],
        "negative": [f"passage: {t.strip()}" for t in triplets_df["negative"].astype(str)],
    })


def save_stage_report(
    stage: str,
    model_name: str,
    config: dict,
    metrics: dict,
    start_time: datetime
) -> None:
    """Guarda reporte de configuración y resultados de la etapa."""
    output_dir = report_dir_for_tag(stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    elapsed = datetime.now(timezone.utc) - start_time
    report = {
        "pipeline": f"finetune_st_{stage}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "config": config,
        "retrieval_metrics": metrics,
        "elapsed_seconds": elapsed.total_seconds()
    }

    with open(output_dir / f"{stage}_training.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Stage: baseline
# ---------------------------------------------------------------------------

def run_baseline(args: argparse.Namespace) -> None:
    """Evalúa el modelo pre-entrenado sin fine-tuning."""
    start_time = datetime.now(timezone.utc)

    print("=" * 70)
    print("  ETAPA 04d: BASELINE (sin fine-tuning)")
    print("=" * 70)
    print(f"\n  Modelo: {args.model}")

    model = SentenceTransformer(args.model)

    print("  Cargando test split...")
    test_df = load_split("test")
    print(f"  Pares de test: {len(test_df):,}")

    print("\n  Evaluando retrieval...")
    metrics = evaluate_model(model, test_df, "baseline", args.model, start_time)

    save_stage_report("baseline", args.model, {}, metrics, start_time)

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage: v1 — Fine-tuning con pares (MultipleNegativesRankingLoss)
# ---------------------------------------------------------------------------

def run_v1(args: argparse.Namespace) -> None:
    """Fine-tuning con pares usando MultipleNegativesRankingLoss."""
    start_time = datetime.now(timezone.utc)
    epochs = args.epochs or 10

    print("=" * 70)
    print("  ETAPA 04d: FINE-TUNING v1 (MultipleNegativesRankingLoss)")
    print("=" * 70)
    print(f"\n  Modelo base:    {args.model}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Salida modelo:  {V1_DIR}")

    model = SentenceTransformer(args.model)

    print("\n  Cargando splits...")
    train_df = load_split("train")
    valid_df = load_split("valid")
    print(f"  Train: {len(train_df):,} pares")
    print(f"  Valid: {len(valid_df):,} pares")

    train_dataset = build_pair_dataset(train_df)
    eval_dataset = build_pair_dataset(valid_df)

    loss = MultipleNegativesRankingLoss(model)

    V1_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(V1_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    print("\n  Iniciando entrenamiento v1...")
    trainer.train()

    print(f"\n  Guardando modelo en {V1_DIR}")
    model.save_pretrained(str(V1_DIR))

    print("\n  Evaluando v1 en test split...")
    test_df = load_split("test")
    metrics = evaluate_model(model, test_df, "v1", str(V1_DIR), start_time)

    config = {
        "base_model": args.model,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "loss": "MultipleNegativesRankingLoss",
        "train_pairs": len(train_df),
        "valid_pairs": len(valid_df),
    }
    save_stage_report("v1", str(V1_DIR), config, metrics, start_time)

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage: mine-negatives — Minería de hard negatives
# ---------------------------------------------------------------------------

def run_mine_negatives(args: argparse.Namespace) -> None:
    """Mina hard negatives desde el modelo v1 entrenado."""
    start_time = datetime.now(timezone.utc)

    print("=" * 70)
    print("  ETAPA 04d: MINERÍA DE HARD NEGATIVES")
    print("=" * 70)

    if not V1_DIR.exists():
        raise FileNotFoundError(
            f"Modelo v1 no encontrado: {V1_DIR}\n"
            "Ejecuta primero: poetry run python src/embeddings/finetune_st.py --stage v1"
        )

    print(f"\n  Cargando modelo v1 desde {V1_DIR}")
    model = SentenceTransformer(str(V1_DIR))

    print("  Cargando train split...")
    train_df = load_split("train")
    print(f"  Pares de entrenamiento: {len(train_df):,}")

    esp_texts = train_df["ESP_normalizado"].astype(str).tolist()
    shi_texts = train_df["SHIWILU_normalizado"].astype(str).tolist()

    queries = [f"query: {t.strip()}" for t in esp_texts]
    passages = [f"passage: {t.strip()}" for t in shi_texts]

    print("\n  Codificando queries (español)...")
    emb_q = model.encode(queries, show_progress_bar=True, convert_to_numpy=True,
                         batch_size=64)
    print("  Codificando passages (shiwilu)...")
    emb_p = model.encode(passages, show_progress_bar=True, convert_to_numpy=True,
                         batch_size=64)

    # Similitud coseno via producto punto sobre vectores normalizados
    emb_q_norm = emb_q / np.linalg.norm(emb_q, axis=1, keepdims=True)
    emb_p_norm = emb_p / np.linalg.norm(emb_p, axis=1, keepdims=True)
    sim_matrix = emb_q_norm @ emb_p_norm.T

    print("\n  Minando hard negatives...")
    negatives = []
    neg_scores = []
    pos_scores = []
    for i in range(len(esp_texts)):
        scores = sim_matrix[i].copy()
        pos_scores.append(float(scores[i]))
        scores[i] = -2.0  # excluir el positivo
        hard_neg_idx = int(np.argmax(scores))
        negatives.append(shi_texts[hard_neg_idx])
        neg_scores.append(float(sim_matrix[i, hard_neg_idx]))

    triplets_df = pd.DataFrame({
        "anchor": esp_texts,
        "positive": shi_texts,
        "negative": negatives,
    })

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    triplets_df.to_csv(TRIPLETS_PATH, index=False, encoding="utf-8-sig")

    elapsed = datetime.now(timezone.utc) - start_time
    pos_scores_arr = np.array(pos_scores)
    neg_scores_arr = np.array(neg_scores)
    margin = pos_scores_arr - neg_scores_arr

    # Guardar reporte de minería
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mining_report = {
        "pipeline": "mine_hard_negatives",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_model": str(V1_DIR),
        "total_triplets": len(triplets_df),
        "positive_similarity": {
            "mean": float(np.mean(pos_scores_arr)),
            "std": float(np.std(pos_scores_arr)),
            "min": float(np.min(pos_scores_arr)),
            "max": float(np.max(pos_scores_arr)),
        },
        "hard_negative_similarity": {
            "mean": float(np.mean(neg_scores_arr)),
            "std": float(np.std(neg_scores_arr)),
            "min": float(np.min(neg_scores_arr)),
            "max": float(np.max(neg_scores_arr)),
        },
        "margin_stats": {
            "mean": float(np.mean(margin)),
            "std": float(np.std(margin)),
            "min": float(np.min(margin)),
            "pct_positive_margin": float(np.mean(margin > 0) * 100),
        },
        "elapsed_seconds": elapsed.total_seconds()
    }

    legacy_v2_dir = REPORTS_DIR / "legacy_v2"
    legacy_v2_dir.mkdir(parents=True, exist_ok=True)
    with open(legacy_v2_dir / "mine_negatives_report.json", "w", encoding="utf-8") as f:
        json.dump(mining_report, f, ensure_ascii=False, indent=2)

    print()
    print("  " + "-" * 50)
    print("  MINERÍA DE HARD NEGATIVES - RESULTADOS")
    print("  " + "-" * 50)
    print(f"    Triplets generados:       {len(triplets_df):,}")
    print(f"    Sim. positivo (media):    {np.mean(pos_scores_arr):.4f}")
    print(f"    Sim. hard neg (media):    {np.mean(neg_scores_arr):.4f}")
    print(f"    Margen medio:             {np.mean(margin):.4f}")
    print(f"    % con margen positivo:    {np.mean(margin > 0) * 100:.1f}%")
    print(f"    Tiempo:                   {elapsed.total_seconds():.2f}s")
    print()
    print("  SALIDAS:")
    print("  " + "-" * 50)
    print(f"    Triplets:  {TRIPLETS_PATH}")
    print(f"    Reporte:   {legacy_v2_dir / 'mine_negatives_report.json'}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage: v2 — Fine-tuning con triplets (TripletLoss)
# ---------------------------------------------------------------------------

def run_v2(args: argparse.Namespace) -> None:
    """Fine-tuning con triplets usando TripletLoss, inicializando desde v1."""
    start_time = datetime.now(timezone.utc)
    epochs = args.epochs or 5

    print("=" * 70)
    print("  ETAPA 04d: FINE-TUNING v2 (TripletLoss)")
    print("=" * 70)

    if not V1_DIR.exists():
        raise FileNotFoundError(
            f"Modelo v1 no encontrado: {V1_DIR}\n"
            "Ejecuta primero: poetry run python src/embeddings/finetune_st.py --stage v1"
        )
    if not TRIPLETS_PATH.exists():
        raise FileNotFoundError(
            f"Triplets no encontrados: {TRIPLETS_PATH}\n"
            "Ejecuta primero: poetry run python src/embeddings/finetune_st.py --stage mine-negatives"
        )

    print(f"\n  Modelo base:    {V1_DIR} (inicializa desde v1)")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Salida modelo:  {V2_DIR}")

    model = SentenceTransformer(str(V1_DIR))

    print("\n  Cargando triplets...")
    triplets_df = pd.read_csv(TRIPLETS_PATH, encoding="utf-8-sig")
    print(f"  Triplets: {len(triplets_df):,}")

    train_dataset = build_triplet_dataset(triplets_df)

    loss = TripletLoss(model)

    V2_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(V2_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print("\n  Iniciando entrenamiento v2...")
    trainer.train()

    print(f"\n  Guardando modelo en {V2_DIR}")
    model.save_pretrained(str(V2_DIR))

    print("\n  Evaluando v2 en test split...")
    test_df = load_split("test")
    metrics = evaluate_model(model, test_df, "v2", str(V2_DIR), start_time)

    config = {
        "base_model": str(V1_DIR),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "loss": "TripletLoss",
        "train_triplets": len(triplets_df),
    }
    save_stage_report("v2", str(V2_DIR), config, metrics, start_time)

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Función principal: despacha a la etapa correspondiente."""
    args = parse_args()

    dispatch = {
        "baseline": run_baseline,
        "v1": run_v1,
        "mine-negatives": run_mine_negatives,
        "v2": run_v2,
    }

    dispatch[args.stage](args)


if __name__ == "__main__":
    main()
