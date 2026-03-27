"""
train_sentence_transformers.py
Generación de embeddings cross-lingual con Sentence Transformers.

Usa un modelo multilingüe pre-entrenado para generar embeddings a nivel de
oración y medir similitud entre pares español-shiwilu.

Uso:
    poetry run python src/embeddings/train_sentence_transformers.py

Salida:
    models/sentence_transformers/embeddings_esp.npy
    models/sentence_transformers/embeddings_shi.npy
    reports/04_embeddings/similarity_scores.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_pre_embeddings" / "dataset_pre_embeddings.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "sentence_transformers"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"

MODEL_NAME = "intfloat/multilingual-e5-small"


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Genera embeddings cross-lingual con Sentence Transformers"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Ruta al CSV del corpus (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Modelo de Sentence Transformers (default: {MODEL_NAME})"
    )
    return parser.parse_args()


def load_corpus(filepath: Path) -> pd.DataFrame:
    """Carga el corpus y retorna DataFrame."""
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    prefix: str
) -> np.ndarray:
    """
    Genera embeddings para una lista de textos.
    
    Para modelos E5, SIEMPRE usar prefijos asimétricos:
    - "query: " para textos de consulta (lo que buscas)
    - "passage: " para textos de documento (donde buscas)
    """
    prefixed_texts = [f"{prefix}{t.strip()}" for t in texts]
    embeddings = model.encode(prefixed_texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def calculate_similarity_scores(
    embeddings_esp: np.ndarray,
    embeddings_shi: np.ndarray
) -> np.ndarray:
    """Calcula similitud coseno entre pares correspondientes."""
    scores = []
    for i in range(len(embeddings_esp)):
        sim = cos_sim(embeddings_esp[i], embeddings_shi[i]).item()
        scores.append(sim)
    return np.array(scores)


def save_outputs(
    df: pd.DataFrame,
    embeddings_esp: np.ndarray,
    embeddings_shi: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    start_time: datetime
) -> None:
    """Guarda embeddings, scores y reportes."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(MODEL_DIR / "embeddings_esp.npy", embeddings_esp)
    np.save(MODEL_DIR / "embeddings_shi.npy", embeddings_shi)
    
    scores_df = pd.DataFrame({
        "pair_id": df["pair_id"],
        "ESP_normalizado": df["ESP_normalizado"],
        "SHIWILU_normalizado": df["SHIWILU_normalizado"],
        "similarity_score": scores
    })
    scores_df.to_csv(REPORTS_DIR / "similarity_scores.csv", index=False, encoding="utf-8-sig")
    
    elapsed = datetime.now(timezone.utc) - start_time
    summary = {
        "pipeline": "train_sentence_transformers",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "total_pairs": len(df),
        "embedding_dim": embeddings_esp.shape[1],
        "similarity_stats": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        },
        "elapsed_seconds": elapsed.total_seconds()
    }
    
    with open(REPORTS_DIR / "sentence_transformers_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(
    df: pd.DataFrame,
    scores: np.ndarray,
    embeddings_esp: np.ndarray,
    model_name: str,
    start_time: datetime
) -> None:
    """Imprime reporte en consola."""
    elapsed = datetime.now(timezone.utc) - start_time
    
    print("\n" + "=" * 70)
    print("  ETAPA 04b: EMBEDDINGS SENTENCE TRANSFORMERS")
    print("=" * 70)
    print()
    print(f"  Modelo:               {model_name}")
    print(f"  Total pares:          {len(df):,}")
    print(f"  Dimensión embeddings: {embeddings_esp.shape[1]}")
    print(f"  Tiempo de ejecución:  {elapsed.total_seconds():.2f}s")
    print()
    print("  ESTADÍSTICAS DE SIMILITUD:")
    print("  " + "-" * 50)
    print(f"    Media:    {np.mean(scores):.4f}")
    print(f"    Std:      {np.std(scores):.4f}")
    print(f"    Min:      {np.min(scores):.4f}")
    print(f"    Max:      {np.max(scores):.4f}")
    print(f"    Mediana:  {np.median(scores):.4f}")
    
    sorted_indices = np.argsort(scores)
    
    print()
    print("  PARES CON MENOR SIMILITUD (candidatos a revisar):")
    print("  " + "-" * 50)
    for i in sorted_indices[:5]:
        row = df.iloc[i]
        print(f"    [{scores[i]:.4f}] {row['ESP_normalizado'][:40]}")
        print(f"             {row['SHIWILU_normalizado'][:40]}")
        print()
    
    print("  PARES CON MAYOR SIMILITUD:")
    print("  " + "-" * 50)
    for i in sorted_indices[-5:][::-1]:
        row = df.iloc[i]
        print(f"    [{scores[i]:.4f}] {row['ESP_normalizado'][:40]}")
        print(f"             {row['SHIWILU_normalizado'][:40]}")
        print()
    
    print("  SALIDAS GENERADAS:")
    print("  " + "-" * 50)
    print(f"    Embeddings ESP:  {MODEL_DIR / 'embeddings_esp.npy'}")
    print(f"    Embeddings SHI:  {MODEL_DIR / 'embeddings_shi.npy'}")
    print(f"    Scores CSV:      {REPORTS_DIR / 'similarity_scores.csv'}")
    print(f"    Resumen JSON:    {REPORTS_DIR / 'sentence_transformers_summary.json'}")
    print("=" * 70)


def main() -> None:
    """Función principal."""
    start_time = datetime.now(timezone.utc)
    args = parse_args()
    
    print("=" * 70)
    print("  Cargando modelo Sentence Transformers...")
    print("=" * 70)
    print(f"\n  Modelo: {args.model}")
    
    model = SentenceTransformer(args.model)
    
    print(f"\n  Cargando corpus: {args.data}")
    df = load_corpus(args.data)
    print(f"  Pares cargados: {len(df):,}")
    
    esp_texts = df["ESP_normalizado"].astype(str).tolist()
    shi_texts = df["SHIWILU_normalizado"].astype(str).tolist()
    
    print("\n  Generando embeddings para español (query)...")
    embeddings_esp = generate_embeddings(model, esp_texts, prefix="query: ")
    
    print("\n  Generando embeddings para shiwilu (passage)...")
    embeddings_shi = generate_embeddings(model, shi_texts, prefix="passage: ")
    
    print("\n  Calculando similitudes...")
    scores = calculate_similarity_scores(embeddings_esp, embeddings_shi)
    
    save_outputs(df, embeddings_esp, embeddings_shi, scores, args.model, start_time)
    print_report(df, scores, embeddings_esp, args.model, start_time)


if __name__ == "__main__":
    main()
