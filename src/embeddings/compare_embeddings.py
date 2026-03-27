"""
compare_embeddings.py
Comparación entre FastText y Sentence Transformers.

Analiza las diferencias en cómo cada método representa el corpus
español-shiwilu y genera un reporte comparativo.

Uso:
    poetry run python src/embeddings/compare_embeddings.py

Salida:
    reports/04_embeddings/comparison_report.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_pre_embeddings" / "dataset_pre_embeddings.csv"
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "models" / "fasttext" / "fasttext.model"
ST_EMBEDDINGS_ESP = PROJECT_ROOT / "models" / "sentence_transformers" / "embeddings_esp.npy"
ST_EMBEDDINGS_SHI = PROJECT_ROOT / "models" / "sentence_transformers" / "embeddings_shi.npy"
ST_SCORES_PATH = PROJECT_ROOT / "reports" / "04_embeddings" / "similarity_scores.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings"


def load_fasttext_model() -> FastText:
    """Carga modelo FastText entrenado."""
    if not FASTTEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo FastText no encontrado: {FASTTEXT_MODEL_PATH}")
    return FastText.load(str(FASTTEXT_MODEL_PATH))


def load_sentence_transformer_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Carga embeddings y scores de Sentence Transformers."""
    if not ST_EMBEDDINGS_ESP.exists():
        raise FileNotFoundError(
            f"Embeddings no encontrados: {ST_EMBEDDINGS_ESP}\n"
            "Ejecuta primero: poetry run python src/embeddings/train_sentence_transformers.py"
        )
    
    embeddings_esp = np.load(ST_EMBEDDINGS_ESP)
    embeddings_shi = np.load(ST_EMBEDDINGS_SHI)
    scores_df = pd.read_csv(ST_SCORES_PATH, encoding="utf-8-sig")
    
    return embeddings_esp, embeddings_shi, scores_df


def fasttext_word_similarity(model: FastText, word1: str, word2: str) -> float:
    """Calcula similitud entre dos palabras con FastText."""
    try:
        return float(model.wv.similarity(word1, word2))
    except KeyError:
        return 0.0


def analyze_sample_pairs(
    ft_model: FastText,
    st_scores_df: pd.DataFrame,
    n_samples: int = 10
) -> list[dict]:
    """Analiza pares de ejemplo con ambos métodos."""
    results = []
    
    sample_indices = np.linspace(0, len(st_scores_df) - 1, n_samples, dtype=int)
    
    for idx in sample_indices:
        row = st_scores_df.iloc[idx]
        esp_text = str(row["ESP_normalizado"])
        shi_text = str(row["SHIWILU_normalizado"])
        st_score = float(row["similarity_score"])
        
        esp_words = esp_text.split()
        shi_words = shi_text.split()
        
        ft_scores = []
        for esp_w in esp_words[:3]:
            for shi_w in shi_words[:3]:
                sim = fasttext_word_similarity(ft_model, esp_w, shi_w)
                ft_scores.append(sim)
        
        ft_avg = np.mean(ft_scores) if ft_scores else 0.0
        
        results.append({
            "pair_id": row["pair_id"],
            "esp": esp_text[:50],
            "shiwilu": shi_text[:50],
            "sentence_transformer_score": st_score,
            "fasttext_avg_word_score": float(ft_avg),
            "difference": st_score - ft_avg
        })
    
    return results


def compare_distributions(st_scores_df: pd.DataFrame) -> dict:
    """Compara distribuciones de scores."""
    scores = st_scores_df["similarity_score"].values
    
    percentiles = [10, 25, 50, 75, 90]
    
    return {
        "sentence_transformers": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "percentiles": {str(p): float(np.percentile(scores, p)) for p in percentiles}
        }
    }


def identify_problematic_pairs(st_scores_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Identifica pares con baja similitud (candidatos a filtrar)."""
    low_sim = st_scores_df[st_scores_df["similarity_score"] < threshold].copy()
    return low_sim.sort_values("similarity_score")


def save_report(
    sample_analysis: list[dict],
    distributions: dict,
    problematic_pairs: pd.DataFrame,
    ft_vocab_size: int,
    st_embedding_dim: int
) -> None:
    """Guarda reporte comparativo."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "fasttext": {
                "type": "word embeddings (morfológico)",
                "vocab_size": ft_vocab_size,
                "trained_on": "corpus propio"
            },
            "sentence_transformers": {
                "type": "sentence embeddings (cross-lingual)",
                "model": "intfloat/multilingual-e5-small",
                "embedding_dim": st_embedding_dim,
                "pretrained": True
            }
        },
        "distributions": distributions,
        "sample_analysis": sample_analysis,
        "problematic_pairs_count": len(problematic_pairs),
        "problematic_pairs_threshold": 0.3
    }
    
    with open(REPORTS_DIR / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    if len(problematic_pairs) > 0:
        problematic_pairs.to_csv(
            REPORTS_DIR / "low_similarity_pairs.csv",
            index=False,
            encoding="utf-8-sig"
        )


def print_report(
    sample_analysis: list[dict],
    distributions: dict,
    problematic_pairs: pd.DataFrame,
    ft_vocab_size: int,
    st_embedding_dim: int
) -> None:
    """Imprime reporte en consola."""
    print("\n" + "=" * 70)
    print("  COMPARACIÓN: FastText vs Sentence Transformers")
    print("=" * 70)
    
    print("\n  MODELOS:")
    print("  " + "-" * 50)
    print(f"    FastText:")
    print(f"      - Tipo: Word embeddings (morfológico)")
    print(f"      - Vocabulario: {ft_vocab_size:,} palabras")
    print(f"      - Entrenado: Corpus propio")
    print()
    print(f"    Sentence Transformers:")
    print(f"      - Tipo: Sentence embeddings (cross-lingual)")
    print(f"      - Modelo: intfloat/multilingual-e5-small")
    print(f"      - Dimensión: {st_embedding_dim}")
    print(f"      - Pre-entrenado: Sí (multilingüe)")
    
    print("\n  DISTRIBUCIÓN DE SIMILITUDES (Sentence Transformers):")
    print("  " + "-" * 50)
    st_stats = distributions["sentence_transformers"]
    print(f"    Media:    {st_stats['mean']:.4f}")
    print(f"    Std:      {st_stats['std']:.4f}")
    print(f"    Min:      {st_stats['min']:.4f}")
    print(f"    Max:      {st_stats['max']:.4f}")
    print(f"    P25:      {st_stats['percentiles']['25']:.4f}")
    print(f"    P50:      {st_stats['percentiles']['50']:.4f}")
    print(f"    P75:      {st_stats['percentiles']['75']:.4f}")
    
    print("\n  ANÁLISIS DE PARES DE EJEMPLO:")
    print("  " + "-" * 50)
    print(f"    {'ST Score':<10} {'FT Avg':<10} {'Diff':<10} Par")
    print("    " + "-" * 60)
    for item in sample_analysis[:5]:
        print(f"    {item['sentence_transformer_score']:<10.4f} "
              f"{item['fasttext_avg_word_score']:<10.4f} "
              f"{item['difference']:<+10.4f} "
              f"{item['esp'][:20]}...")
    
    print(f"\n  PARES PROBLEMÁTICOS (similitud < 0.3): {len(problematic_pairs)}")
    if len(problematic_pairs) > 0:
        print("  " + "-" * 50)
        for _, row in problematic_pairs.head(3).iterrows():
            print(f"    [{row['similarity_score']:.4f}] {row['ESP_normalizado'][:35]}")
            print(f"              {row['SHIWILU_normalizado'][:35]}")
    
    print("\n  INTERPRETACIÓN:")
    print("  " + "-" * 50)
    print("    - FastText captura similitud MORFOLÓGICA (subpalabras)")
    print("    - Sentence Transformers captura similitud SEMÁNTICA cross-lingual")
    print("    - Pares con bajo score ST son candidatos a revisar/filtrar")
    print("    - Ambos métodos son complementarios para el pipeline NMT")
    
    print("\n  SALIDAS GENERADAS:")
    print("  " + "-" * 50)
    print(f"    Reporte: {REPORTS_DIR / 'comparison_report.json'}")
    if len(problematic_pairs) > 0:
        print(f"    Pares problemáticos: {REPORTS_DIR / 'low_similarity_pairs.csv'}")
    print("=" * 70)


def main() -> None:
    """Función principal."""
    print("=" * 70)
    print("  Cargando modelos y datos...")
    print("=" * 70)
    
    print("\n  Cargando FastText...")
    ft_model = load_fasttext_model()
    ft_vocab_size = len(ft_model.wv.key_to_index)
    
    print("  Cargando Sentence Transformers embeddings...")
    embeddings_esp, embeddings_shi, st_scores_df = load_sentence_transformer_data()
    st_embedding_dim = embeddings_esp.shape[1]
    
    print(f"\n  FastText vocabulario: {ft_vocab_size:,}")
    print(f"  ST embedding dim: {st_embedding_dim}")
    print(f"  Total pares: {len(st_scores_df):,}")
    
    print("\n  Analizando pares de ejemplo...")
    sample_analysis = analyze_sample_pairs(ft_model, st_scores_df)
    
    print("  Comparando distribuciones...")
    distributions = compare_distributions(st_scores_df)
    
    print("  Identificando pares problemáticos...")
    problematic_pairs = identify_problematic_pairs(st_scores_df)
    
    save_report(sample_analysis, distributions, problematic_pairs, ft_vocab_size, st_embedding_dim)
    print_report(sample_analysis, distributions, problematic_pairs, ft_vocab_size, st_embedding_dim)


if __name__ == "__main__":
    main()
