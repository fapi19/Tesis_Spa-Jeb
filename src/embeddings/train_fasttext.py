"""
train_fasttext.py
Entrenamiento de embeddings FastText (Skip-Gram) para corpus español-shiwilu.

Entrena un modelo FastText desde cero usando el corpus paralelo,
concatenando ambos idiomas en un único espacio de embeddings.

Uso:
    poetry run python src/embeddings/train_fasttext.py
    poetry run python src/embeddings/train_fasttext.py --data ruta/al/corpus.csv

Salida:
    models/fasttext/fasttext.model  - Modelo completo (recargable)
    models/fasttext/fasttext.vec    - Vectores en formato word2vec
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from gensim.models import FastText

from preprocess import load_and_tokenize
from utils import get_similar_words, get_vocab_size, get_vector_size

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_pre_embeddings" / "dataset_pre_embeddings.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "fasttext"
MODEL_PATH = MODEL_DIR / "fasttext.model"
VECTORS_PATH = MODEL_DIR / "fasttext.vec"

FASTTEXT_CONFIG = {
    "vector_size": 50,
    "window": 3,
    "min_count": 1,
    "workers": 4,
    "sg": 1,  # Skip-Gram
}


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena embeddings FastText (Skip-Gram) para corpus español-shiwilu"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Ruta al CSV del corpus (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=FASTTEXT_CONFIG["vector_size"],
        help=f"Dimensión de los vectores (default: {FASTTEXT_CONFIG['vector_size']})"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=FASTTEXT_CONFIG["window"],
        help=f"Tamaño de ventana de contexto (default: {FASTTEXT_CONFIG['window']})"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=FASTTEXT_CONFIG["min_count"],
        help=f"Frecuencia mínima para incluir palabra (default: {FASTTEXT_CONFIG['min_count']})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de epochs de entrenamiento (default: 50)"
    )
    return parser.parse_args()


def train_model(sentences: list[list[str]], config: dict, epochs: int) -> FastText:
    """
    Entrena modelo FastText con el corpus tokenizado.
    
    Args:
        sentences: Lista de oraciones tokenizadas
        config: Configuración del modelo
        epochs: Número de epochs
    
    Returns:
        Modelo FastText entrenado
    """
    print(f"\n  Entrenando modelo FastText (Skip-Gram)...")
    print(f"    vector_size: {config['vector_size']}")
    print(f"    window: {config['window']}")
    print(f"    min_count: {config['min_count']}")
    print(f"    epochs: {epochs}")
    print(f"    sg: {config['sg']} (Skip-Gram)")
    
    model = FastText(
        sentences=sentences,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        workers=config["workers"],
        sg=config["sg"],
        epochs=epochs,
    )
    
    return model


def save_model(model: FastText) -> None:
    """Guarda modelo y vectores."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model.save(str(MODEL_PATH))
    print(f"\n  Modelo guardado:   {MODEL_PATH}")
    
    model.wv.save_word2vec_format(str(VECTORS_PATH))
    print(f"  Vectores guardados: {VECTORS_PATH}")


def validate_model(model: FastText) -> None:
    """Valida modelo con palabras de ejemplo."""
    print("\n  VALIDACIÓN - Palabras similares:")
    print("  " + "-" * 50)
    
    test_words_esp = ["hola", "agua", "casa", "bueno"]
    test_words_shi = ["ma'pu'sin", "paker'", "tekker'", "den"]
    
    for word in test_words_esp + test_words_shi:
        try:
            similar = get_similar_words(model, word, topn=5)
            print(f"    {word}:")
            for w, s in similar:
                print(f"      - {w} [{s:.4f}]")
        except KeyError:
            print(f"    {word}: [no encontrada en vocabulario]")


def print_report(
    sentences: list[list[str]],
    model: FastText,
    data_path: Path,
    start_time: datetime
) -> None:
    """Imprime reporte de entrenamiento."""
    elapsed = datetime.now(timezone.utc) - start_time
    total_tokens = sum(len(s) for s in sentences)
    
    print("\n" + "=" * 70)
    print("  ETAPA 04: ENTRENAMIENTO FastText (Skip-Gram)")
    print("=" * 70)
    print()
    print(f"  Datos de entrada:     {data_path}")
    print(f"  Total oraciones:      {len(sentences):,}")
    print(f"  Total tokens:         {total_tokens:,}")
    print(f"  Vocabulario:          {get_vocab_size(model):,} palabras")
    print(f"  Dimensión vectores:   {get_vector_size(model)}")
    print(f"  Tiempo de ejecución:  {elapsed.total_seconds():.2f}s")
    
    validate_model(model)
    
    print()
    print("  SALIDAS GENERADAS:")
    print("  " + "-" * 50)
    print(f"    Modelo:   {MODEL_PATH}")
    print(f"    Vectores: {VECTORS_PATH}")
    print("=" * 70)


def main() -> None:
    """Función principal de entrenamiento."""
    start_time = datetime.now(timezone.utc)
    args = parse_args()
    
    print("=" * 70)
    print("  Cargando y tokenizando corpus...")
    print("=" * 70)
    print(f"\n  Archivo: {args.data}")
    
    sentences = load_and_tokenize(args.data)
    print(f"  Oraciones cargadas: {len(sentences):,}")
    
    config = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "workers": FASTTEXT_CONFIG["workers"],
        "sg": FASTTEXT_CONFIG["sg"],
    }
    
    model = train_model(sentences, config, args.epochs)
    save_model(model)
    print_report(sentences, model, args.data, start_time)


if __name__ == "__main__":
    main()
