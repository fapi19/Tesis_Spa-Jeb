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
from typing import Literal

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"
Direction = Literal["esp_to_shi", "shi_to_esp"]


def report_dir_for_tag(tag: str, reports_dir: Path | None = None) -> Path:
    root = reports_dir or REPORTS_DIR
    base_tag = tag
    for suffix in ("_esp_to_shi", "_shi_to_esp"):
        if base_tag.endswith(suffix):
            base_tag = base_tag.removesuffix(suffix)
            break
    if base_tag == "baseline":
        return root / "baseline"
    if base_tag == "v1":
        return root / "v1"
    if base_tag == "v2":
        return root / "legacy_v2"
    if base_tag == "v2_hn_controlled":
        return root / "v2_hn_controlled"
    if base_tag == "v2_hn_controlled_hard":
        return root / "v2_hn_controlled_hard"
    return root / "experiments" / base_tag


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
    parser.add_argument(
        "--direction",
        choices=["esp_to_shi", "shi_to_esp"],
        default="esp_to_shi",
        help="Dirección de retrieval a evaluar (default: esp_to_shi)"
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
    batch_size: int = 64,
    direction: Direction = "esp_to_shi",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera embeddings con prefijos E5 asimétricos.

    En cada dirección, query representa lo que se busca y passage el corpus
    candidato sobre el que se rankea.
    """
    if direction == "esp_to_shi":
        query_texts = esp_texts
        passage_texts = shi_texts
    elif direction == "shi_to_esp":
        query_texts = shi_texts
        passage_texts = esp_texts
    else:
        raise ValueError(f"Dirección no soportada: {direction}")

    queries = [f"query: {t.strip()}" for t in query_texts]
    passages = [f"passage: {t.strip()}" for t in passage_texts]

    emb_query = model.encode(queries, show_progress_bar=True, convert_to_numpy=True,
                             batch_size=batch_size)
    emb_passage = model.encode(passages, show_progress_bar=True, convert_to_numpy=True,
                               batch_size=batch_size)
    return emb_query, emb_passage


def build_positive_indices(df: pd.DataFrame) -> list[set[int]]:
    """Build multi-positive retrieval targets from group_id when available."""
    if "group_id" not in df.columns:
        return [{i} for i in range(len(df))]

    group_ids = df["group_id"].astype(str).tolist()
    group_to_indices: dict[str, set[int]] = {}
    for idx, group_id in enumerate(group_ids):
        group_to_indices.setdefault(group_id, set()).add(idx)
    return [group_to_indices[group_id] for group_id in group_ids]


def first_positive_rank(sorted_indices: np.ndarray, positives: set[int]) -> int:
    for rank, idx in enumerate(sorted_indices, start=1):
        if int(idx) in positives:
            return rank
    raise ValueError("No positive candidate found in ranked list.")


def compute_retrieval_metrics(
    emb_query: np.ndarray,
    emb_passage: np.ndarray,
    positive_indices: list[set[int]],
) -> dict:
    """
    Calcula métricas de retrieval bilingüe.

    Para cada query_i, rankea todos los passages por similitud coseno. Si hay
    group_id, cualquier fila del mismo grupo cuenta como positivo válido.

    Retorna dict con recall@1, recall@5, mrr y mean_rank.
    """
    sim_matrix = cos_sim(emb_query, emb_passage).numpy()
    n = sim_matrix.shape[0]

    ranks = []
    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)
        rank = first_positive_rank(sorted_indices, positive_indices[i])
        ranks.append(rank)

    ranks = np.array(ranks)

    recall_at_1 = float(np.mean(ranks <= 1))
    recall_at_5 = float(np.mean(ranks <= 5))
    recall_at_10 = float(np.mean(ranks <= 10))
    mrr = float(np.mean(1.0 / ranks))
    mean_rank = float(np.mean(ranks))

    return {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "total_pairs": n,
        "multi_positive": True,
        "avg_positives_per_query": float(np.mean([len(x) for x in positive_indices])),
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
    positive_indices: list[set[int]],
    direction: Direction = "esp_to_shi",
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
    if direction == "esp_to_shi":
        query_column = "ESP_normalizado"
        target_column = "SHIWILU_normalizado"
    elif direction == "shi_to_esp":
        query_column = "SHIWILU_normalizado"
        target_column = "ESP_normalizado"
    else:
        raise ValueError(f"Dirección no soportada: {direction}")

    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)
        positives = positive_indices[i]
        rank = first_positive_rank(sorted_indices, positives)
        correct_score = float(max(scores[list(positives)]))
        top1_idx = int(sorted_indices[0])
        top1_score = float(scores[top1_idx])
        top1_is_positive = top1_idx in positives

        row_data = df.iloc[i]
        rows.append({
            "pair_id": row_data["pair_id"],
            "group_id": row_data["group_id"] if "group_id" in df.columns else row_data["pair_id"],
            "rank": rank,
            "correct_score": correct_score,
            "top1_score": top1_score,
            "top1_is_positive": top1_is_positive,
            "positive_count": len(positives),
            "direction": direction,
            "query_text": row_data[query_column],
            "target_correct": row_data[target_column],
            "target_top1": (
                "(grupo correcto)"
                if top1_is_positive
                else df.iloc[top1_idx][target_column]
            ),
            "esp": row_data["ESP_normalizado"],
            "shiwilu_correct": row_data["SHIWILU_normalizado"],
            "shiwilu_top1": (
                "(grupo correcto)"
                if top1_is_positive
                else df.iloc[top1_idx]["SHIWILU_normalizado"]
            ),
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
    reports_dir: Path | None = None,
    direction: Direction = "esp_to_shi",
) -> Path:
    """Guarda reporte JSON de retrieval."""
    output_dir = report_dir_for_tag(tag, reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elapsed = datetime.now(timezone.utc) - start_time
    report = {
        "pipeline": "evaluate_retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "tag": tag,
        "direction": direction,
        "metrics": metrics,
        "elapsed_seconds": elapsed.total_seconds()
    }

    output_path = output_dir / f"{tag}_retrieval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path


def save_qualitative_csv(
    qual_df: pd.DataFrame,
    tag: str,
    reports_dir: Path | None = None
) -> Path:
    """Guarda análisis cualitativo como CSV."""
    output_dir = report_dir_for_tag(tag, reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{tag}_qualitative.csv"
    qual_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def evaluate_model(
    model: SentenceTransformer,
    df: pd.DataFrame,
    tag: str,
    model_name: str,
    start_time: datetime | None = None,
    reports_dir: Path | None = None,
    batch_size: int = 64,
    direction: Direction = "esp_to_shi",
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
                                          batch_size=batch_size,
                                          direction=direction)
    positive_indices = build_positive_indices(df)

    metrics = compute_retrieval_metrics(emb_query, emb_passage, positive_indices)
    metrics["direction"] = direction

    qual_df = build_qualitative_analysis(df, emb_query, emb_passage,
                                         positive_indices, direction=direction)

    report_path = save_retrieval_report(metrics, tag, model_name, start_time,
                                        reports_dir, direction=direction)
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
    print(f"    Recall@10:    {metrics['recall@10']:.4f}")
    print(f"    MRR:          {metrics['mrr']:.4f}")
    print(f"    Mean Rank:    {metrics['mean_rank']:.1f}")
    print(f"    Positivos/q:  {metrics['avg_positives_per_query']:.2f}")

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

    base_tag = args.tag or args.split
    tag = f"{base_tag}_{args.direction}"

    print("=" * 70)
    print("  EVALUACIÓN DE RETRIEVAL BILINGÜE")
    print("=" * 70)

    model_path = args.model
    if Path(model_path).exists():
        model_name = str(Path(model_path).resolve())
    else:
        model_name = model_path

    print(f"\n  Modelo: {model_name}")
    print(f"  Dirección: {args.direction}")
    model = SentenceTransformer(model_path)

    print(f"  Cargando split: {args.split}")
    df = load_split(args.split)
    print(f"  Pares: {len(df):,}")

    print("\n  Evaluando retrieval...")
    metrics = evaluate_model(model, df, tag, model_name, start_time,
                             direction=args.direction)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
