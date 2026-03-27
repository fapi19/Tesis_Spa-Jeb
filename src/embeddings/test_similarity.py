"""
test_similarity.py
Script interactivo para probar similitudes con ambos modelos.

Uso:
    poetry run python src/embeddings/test_similarity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FASTTEXT_PATH = PROJECT_ROOT / "models" / "fasttext" / "fasttext.model"
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "03_pre_embeddings" / "dataset_pre_embeddings.csv"


def load_models():
    """Carga ambos modelos y prepara vocabulario para ST."""
    print("Cargando modelos...")
    
    print("  - FastText...")
    ft = FastText.load(str(FASTTEXT_PATH))
    
    print("  - Sentence Transformers...")
    st = SentenceTransformer("intfloat/multilingual-e5-small")
    
    print("  - Construyendo vocabulario para ST...")
    vocab, vocab_embeddings = build_st_vocabulary(st)
    
    print(f"Listo! ({len(vocab):,} palabras en vocabulario ST)\n")
    return ft, st, vocab, vocab_embeddings


def build_st_vocabulary(model: SentenceTransformer) -> tuple[list[str], np.ndarray]:
    """Construye vocabulario y embeddings para búsqueda con ST."""
    df = pd.read_csv(CORPUS_PATH, encoding="utf-8-sig")
    
    words = set()
    for col in ["ESP_normalizado", "SHIWILU_normalizado"]:
        for text in df[col].dropna():
            words.update(str(text).split())
    
    vocab = sorted(list(words))
    prefixed = [f"passage: {w.strip()}" for w in vocab]
    embeddings = model.encode(prefixed, show_progress_bar=False, convert_to_numpy=True)
    
    return vocab, embeddings


def test_fasttext(model: FastText, word: str, topn: int = 10):
    """Muestra palabras similares con FastText."""
    print(f"\n{'='*50}")
    print(f"FASTTEXT - Similares a: {word}")
    print(f"{'='*50}")
    
    try:
        results = model.wv.most_similar(word, topn=topn)
        for w, score in results:
            print(f"  {w:<25} {score:.4f}")
    except Exception as e:
        print(f"  Error: {e}")


def test_st_word(model: SentenceTransformer, word: str, vocab: list[str], vocab_embeddings: np.ndarray, topn: int = 10):
    """Muestra palabras similares con Sentence Transformers."""
    print(f"\n{'='*50}")
    print(f"SENTENCE TRANSFORMERS - Similares a: {word}")
    print(f"{'='*50}")
    
    word_emb = model.encode(f"query: {word.strip()}", convert_to_numpy=True)
    
    similarities = cos_sim(word_emb, vocab_embeddings)[0].numpy()
    
    top_indices = np.argsort(similarities)[::-1][:topn + 1]
    
    for idx in top_indices:
        w = vocab[idx]
        if w.lower() != word.lower():
            print(f"  {w:<25} {similarities[idx]:.4f}")


def test_sentence_transformer(model: SentenceTransformer, text1: str, text2: str):
    """Calcula similitud entre dos textos (text1=query, text2=passage)."""
    print(f"\n{'='*50}")
    print(f"SENTENCE TRANSFORMERS - Similitud")
    print(f"{'='*50}")
    print(f"  Query:   {text1}")
    print(f"  Passage: {text2}")
    
    emb1 = model.encode(f"query: {text1.strip()}", convert_to_numpy=True)
    emb2 = model.encode(f"passage: {text2.strip()}", convert_to_numpy=True)
    
    score = cos_sim(emb1, emb2).item()
    print(f"  Score:   {score:.4f}")
    
    return score


def main():
    ft_model, st_model, vocab, vocab_emb = load_models()
    
    print("=" * 50)
    print("PRUEBAS DE EMBEDDINGS")
    print("=" * 50)
    print("Comandos:")
    print("  ft <palabra>           - Similares con FastText")
    print("  stw <palabra>          - Similares con Sentence Transformers")
    print("  st <texto1> | <texto2> - Similitud entre textos (ST)")
    print("  ambos <palabra>        - Comparar FT y ST lado a lado")
    print("  q                      - Salir")
    print("=" * 50)
    
    while True:
        try:
            entrada = input("\n> ").strip()
            
            if not entrada:
                continue
            
            if entrada.lower() == "q":
                print("Saliendo...")
                break
            
            if entrada.lower().startswith("ft "):
                word = entrada[3:].strip()
                test_fasttext(ft_model, word)
            
            elif entrada.lower().startswith("stw "):
                word = entrada[4:].strip()
                test_st_word(st_model, word, vocab, vocab_emb)
            
            elif entrada.lower().startswith("ambos "):
                word = entrada[6:].strip()
                test_fasttext(ft_model, word)
                test_st_word(st_model, word, vocab, vocab_emb)
            
            elif entrada.lower().startswith("st "):
                parts = entrada[3:].split("|")
                if len(parts) != 2:
                    print("Formato: st <texto1> | <texto2>")
                    continue
                text1 = parts[0].strip()
                text2 = parts[1].strip()
                test_sentence_transformer(st_model, text1, text2)
            
            else:
                print("Comando no reconocido. Usa 'ft', 'stw', 'st', 'ambos' o 'q'")
        
        except KeyboardInterrupt:
            print("\nSaliendo...")
            break


if __name__ == "__main__":
    main()
