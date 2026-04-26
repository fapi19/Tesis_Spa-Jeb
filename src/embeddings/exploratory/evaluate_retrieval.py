"""
evaluate_retrieval.py
Evaluación de retrieval bilingüe para modelos Sentence Transformers.

Calcula métricas de retrieval (Recall@1, Recall@5, MRR, Mean Rank)
y genera reportes JSON + análisis cualitativo CSV.

Uso como módulo:
    from evaluate_retrieval import evaluate_model

Uso standalone:
    poetry run python src/embeddings/evaluate_retrieval.py \\
        --model intfloat/multilingual-e5-small --split test --tag baseline
    poetry run python src/embeddings/evaluate_retrieval.py \\
        --model models/sentence_transformers/finetuned_v1 --split test --tag v1

Salida:
    reports/04_embeddings/{tag}_retrieval.json
    reports/04_embeddings/{tag}_qualitative.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evalúa retrieval bilingüe de un modelo Sentence Transformers"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Nombre o ruta del modelo Sentence Transformers"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default="test",
        help="Split a evaluar (default: test)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Etiqueta para los archivos de salida (default: nombre del split)"
    )
    return parser.parse_args()


def load_split(split_name: str, splits_dir: Path | None = None) -> pd.DataFrame:
    """Carga un split CSV."""
    splits_dir = splits_dir or SPLITS_DIR
    filepath = splits_dir / f"{split_name}.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Split no encontrado: {filepath}\n"
            "Ejecuta primero: poetry run python -m src.embeddings.preprocess_embeddings"
        )
    return pd.read_csv(filepath, encoding="utf-8-sig")


def encode_pairs(
    model: SentenceTransformer,
    esp_texts: list[str],
    shi_texts: list[str],
    batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera embeddings con prefijos E5 asimétricos.

    ESP → "query: ..." (lo que buscamos)
    SHI → "passage: ..." (el corpus donde buscamos)
    """
    queries = [f"query: {t.strip()}" for t in esp_texts]
    passages = [f"passage: {t.strip()}" for t in shi_texts]

    emb_query = model.encode(queries, show_progress_bar=True, convert_to_numpy=True,
                             batch_size=batch_size)
    emb_passage = model.encode(passages, show_progress_bar=True, convert_to_numpy=True,
                               batch_size=batch_size)
    return emb_query, emb_passage


def compute_retrieval_metrics(
    emb_query: np.ndarray,
    emb_passage: np.ndarray
) -> dict:
    """
    Calcula métricas de retrieval bilingüe.

    Para cada query_i, rankea todos los passages por similitud coseno.
    El ground truth es que query_i corresponde a passage_i.

    Retorna dict con recall@1, recall@5, mrr y mean_rank.
    """
    sim_matrix = cos_sim(emb_query, emb_passage).numpy()
    n = sim_matrix.shape[0]

    ranks = []
    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    recall_at_1 = float(np.mean(ranks <= 1))
    recall_at_5 = float(np.mean(ranks <= 5))
    mrr = float(np.mean(1.0 / ranks))
    mean_rank = float(np.mean(ranks))

    return {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "total_pairs": n,
        "rank_distribution": {
            "rank_1": int(np.sum(ranks == 1)),
            "rank_2_5": int(np.sum((ranks >= 2) & (ranks <= 5))),
            "rank_6_10": int(np.sum((ranks >= 6) & (ranks <= 10))),
            "rank_11_50": int(np.sum((ranks >= 11) & (ranks <= 50))),
            "rank_51_plus": int(np.sum(ranks > 50)),
        }
    }


def build_qualitative_analysis(
    df: pd.DataFrame,
    emb_query: np.ndarray,
    emb_passage: np.ndarray,
    n_best: int = 20,
    n_worst: int = 20
) -> pd.DataFrame:
    """
    Genera análisis cualitativo: los mejores y peores retrievals.

    Para cada query, muestra qué rank obtuvo su passage correcto,
    el score del correcto, y qué passage fue rankeado primero (si no es el correcto).
    """
    sim_matrix = cos_sim(emb_query, emb_passage).numpy()
    n = sim_matrix.shape[0]

    rows = []
    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)
        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        correct_score = float(scores[i])
        top1_idx = int(sorted_indices[0])
        top1_score = float(scores[top1_idx])

        row_data = df.iloc[i]
        rows.append({
            "pair_id": row_data["pair_id"],
            "rank": rank,
            "correct_score": correct_score,
            "top1_score": top1_score,
            "esp": row_data["ESP_normalizado"],
            "shiwilu_correct": row_data["SHIWILU_normalizado"],
            "shiwilu_top1": df.iloc[top1_idx]["SHIWILU_normalizado"] if top1_idx != i else "(correcto)",
        })

    result_df = pd.DataFrame(rows).sort_values("rank")

    best = result_df.head(n_best).copy()
    best["category"] = "best"
    worst = result_df.tail(n_worst).copy()
    worst["category"] = "worst"

    return pd.concat([best, worst], ignore_index=True)


def save_retrieval_report(
    metrics: dict,
    tag: str,
    model_name: str,
    start_time: datetime,
    reports_dir: Path | None = None
) -> Path:
    """Guarda reporte JSON de retrieval."""
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    elapsed = datetime.now(timezone.utc) - start_time
    report = {
        "pipeline": "evaluate_retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "tag": tag,
        "metrics": metrics,
        "elapsed_seconds": elapsed.total_seconds()
    }

    output_path = reports_dir / f"{tag}_retrieval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path


def save_qualitative_csv(
    qual_df: pd.DataFrame,
    tag: str,
    reports_dir: Path | None = None
) -> Path:
    """Guarda análisis cualitativo como CSV."""
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / f"{tag}_qualitative.csv"
    qual_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def evaluate_model(
    model: SentenceTransformer,
    df: pd.DataFrame,
    tag: str,
    model_name: str,
    start_time: datetime | None = None,
    reports_dir: Path | None = None,
    batch_size: int = 64
) -> dict:
    """
    Evalúa un modelo completo sobre un DataFrame con pares.

    Genera embeddings, calcula métricas, guarda reportes.
    Retorna las métricas.
    """
    start_time = start_time or datetime.now(timezone.utc)
    reports_dir = reports_dir or REPORTS_DIR

    esp_texts = df["ESP_normalizado"].astype(str).tolist()
    shi_texts = df["SHIWILU_normalizado"].astype(str).tolist()

    emb_query, emb_passage = encode_pairs(model, esp_texts, shi_texts,
                                          batch_size=batch_size)

    metrics = compute_retrieval_metrics(emb_query, emb_passage)

    qual_df = build_qualitative_analysis(df, emb_query, emb_passage)

    report_path = save_retrieval_report(metrics, tag, model_name, start_time,
                                        reports_dir)
    qual_path = save_qualitative_csv(qual_df, tag, reports_dir)

    print_metrics(metrics, tag, model_name, report_path, qual_path)

    return metrics


def print_metrics(
    metrics: dict,
    tag: str,
    model_name: str,
    report_path: Path,
    qual_path: Path
) -> None:
    """Imprime métricas de retrieval en consola."""
    print()
    print("  " + "-" * 50)
    print(f"  RETRIEVAL [{tag}] — {model_name}")
    print("  " + "-" * 50)
    print(f"    Total pares:  {metrics['total_pairs']:,}")
    print(f"    Recall@1:     {metrics['recall@1']:.4f}")
    print(f"    Recall@5:     {metrics['recall@5']:.4f}")
    print(f"    MRR:          {metrics['mrr']:.4f}")
    print(f"    Mean Rank:    {metrics['mean_rank']:.1f}")

    dist = metrics["rank_distribution"]
    print()
    print("    Distribución de ranks:")
    print(f"      Rank 1:      {dist['rank_1']}")
    print(f"      Rank 2-5:    {dist['rank_2_5']}")
    print(f"      Rank 6-10:   {dist['rank_6_10']}")
    print(f"      Rank 11-50:  {dist['rank_11_50']}")
    print(f"      Rank 51+:    {dist['rank_51_plus']}")

    print()
    print(f"    Reporte:      {report_path}")
    print(f"    Cualitativo:  {qual_path}")


def main() -> None:
    """Función principal para evaluación standalone."""
    start_time = datetime.now(timezone.utc)
    args = parse_args()

    tag = args.tag or args.split

    print("=" * 70)
    print("  EVALUACIÓN DE RETRIEVAL BILINGÜE")
    print("=" * 70)

    model_path = args.model
    if Path(model_path).exists():
        model_name = str(Path(model_path).resolve())
    else:
        model_name = model_path

    print(f"\n  Modelo: {model_name}")
    model = SentenceTransformer(model_path)

    print(f"  Cargando split: {args.split}")
    df = load_split(args.split)
    print(f"  Pares: {len(df):,}")

    print("\n  Evaluando retrieval...")
    metrics = evaluate_model(model, df, tag, model_name, start_time)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
