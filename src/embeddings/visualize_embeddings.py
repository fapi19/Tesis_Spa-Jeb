"""
visualize_embeddings.py
Visualización 2D de embeddings FastText y Sentence Transformers usando t-SNE.

Uso:
    poetry run python src/embeddings/visualize_embeddings.py
    poetry run python src/embeddings/visualize_embeddings.py --model ft
    poetry run python src/embeddings/visualize_embeddings.py --model st
    poetry run python src/embeddings/visualize_embeddings.py --highlight agua,casa,hola
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import FastText
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FASTTEXT_PATH = PROJECT_ROOT / "models" / "fasttext" / "fasttext.model"
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "03_pre_embeddings" / "dataset_pre_embeddings.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "04_embeddings"


def load_corpus() -> tuple[list[str], list[str]]:
    """Carga palabras únicas del corpus separadas por idioma."""
    df = pd.read_csv(CORPUS_PATH, encoding="utf-8-sig")

    esp_words = set()
    shi_words = set()

    for text in df["ESP_normalizado"].dropna():
        esp_words.update(str(text).split())
    for text in df["SHIWILU_normalizado"].dropna():
        shi_words.update(str(text).split())

    overlap = esp_words & shi_words
    esp_words -= overlap
    shi_words -= overlap

    return sorted(list(esp_words)), sorted(list(shi_words))


def get_fasttext_data(esp_words: list[str], shi_words: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Obtiene vectores FastText para todas las palabras."""
    print("  Cargando modelo FastText...")
    model = FastText.load(str(FASTTEXT_PATH))

    all_words = esp_words + shi_words
    labels = ["ESP"] * len(esp_words) + ["SHI"] * len(shi_words)

    vectors = []
    valid_words = []
    valid_labels = []

    for word, label in zip(all_words, labels):
        try:
            vec = model.wv[word]
            vectors.append(vec)
            valid_words.append(word)
            valid_labels.append(label)
        except KeyError:
            continue

    return np.array(vectors), valid_words, valid_labels


def get_st_data(esp_words: list[str], shi_words: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Obtiene vectores Sentence Transformers para todas las palabras."""
    print("  Cargando Sentence Transformers...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    all_words = esp_words + shi_words
    labels = ["ESP"] * len(esp_words) + ["SHI"] * len(shi_words)

    prefixed = [f"passage: {w.strip()}" for w in all_words]
    vectors = model.encode(prefixed, show_progress_bar=True, convert_to_numpy=True)

    return vectors, all_words, labels


def apply_tsne(vectors: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """Reduce dimensionalidad con t-SNE."""
    effective_perplexity = min(perplexity, len(vectors) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=effective_perplexity, max_iter=1000)
    return tsne.fit_transform(vectors)


def plot_embeddings(
    coords: np.ndarray,
    words: list[str],
    labels: list[str],
    title: str,
    highlight_words: list[str] | None = None,
    output_path: Path | None = None,
):
    """Genera el gráfico 2D de embeddings."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    esp_mask = np.array([l == "ESP" for l in labels])
    shi_mask = np.array([l == "SHI" for l in labels])

    ax.scatter(
        coords[esp_mask, 0], coords[esp_mask, 1],
        c="#4fc3f7", alpha=0.5, s=12, label="Español", edgecolors="none",
    )
    ax.scatter(
        coords[shi_mask, 0], coords[shi_mask, 1],
        c="#ff8a65", alpha=0.5, s=12, label="Shiwilu", edgecolors="none",
    )

    if highlight_words:
        for i, word in enumerate(words):
            if word in highlight_words:
                color = "#4fc3f7" if labels[i] == "ESP" else "#ff8a65"
                ax.scatter(
                    coords[i, 0], coords[i, 1],
                    c=color, s=120, edgecolors="white", linewidths=1.5, zorder=5,
                )
                ax.annotate(
                    word,
                    (coords[i, 0], coords[i, 1]),
                    fontsize=9, fontweight="bold", color="white",
                    xytext=(8, 8), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a", edgecolor=color, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                )
    else:
        step = max(1, len(words) // 60)
        for i in range(0, len(words), step):
            color = "#4fc3f7" if labels[i] == "ESP" else "#ff8a65"
            ax.annotate(
                words[i],
                (coords[i, 0], coords[i, 1]),
                fontsize=6, color=color, alpha=0.7,
                xytext=(4, 4), textcoords="offset points",
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4fc3f7", markersize=10, label="Español", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff8a65", markersize=10, label="Shiwilu", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=15)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Guardado: {output_path}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizar embeddings")
    parser.add_argument("--model", choices=["ft", "st", "ambos"], default="ambos", help="Modelo a visualizar")
    parser.add_argument("--highlight", type=str, default=None, help="Palabras a resaltar, separadas por coma")
    parser.add_argument("--max-words", type=int, default=800, help="Máximo de palabras por idioma")
    return parser.parse_args()


def main():
    args = parse_args()

    highlight = args.highlight.split(",") if args.highlight else None

    print("Cargando corpus...")
    esp_words, shi_words = load_corpus()

    esp_words = esp_words[:args.max_words]
    shi_words = shi_words[:args.max_words]
    print(f"  Español: {len(esp_words)} palabras | Shiwilu: {len(shi_words)} palabras")

    if args.model in ("ft", "ambos"):
        print("\n--- FastText ---")
        vectors, words, labels = get_fasttext_data(esp_words, shi_words)
        print(f"  Vectores: {vectors.shape}")
        print("  Aplicando t-SNE...")
        coords = apply_tsne(vectors)
        plot_embeddings(
            coords, words, labels,
            "FastText (Skip-Gram) — Espacio de Embeddings",
            highlight_words=highlight,
            output_path=OUTPUT_DIR / "tsne_fasttext.png",
        )

    if args.model in ("st", "ambos"):
        print("\n--- Sentence Transformers ---")
        vectors, words, labels = get_st_data(esp_words, shi_words)
        print(f"  Vectores: {vectors.shape}")
        print("  Aplicando t-SNE...")
        coords = apply_tsne(vectors)
        plot_embeddings(
            coords, words, labels,
            "Sentence Transformers (multilingual-e5-small) — Espacio de Embeddings",
            highlight_words=highlight,
            output_path=OUTPUT_DIR / "tsne_sentence_transformers.png",
        )

    print("\nListo!")


if __name__ == "__main__":
    main()
