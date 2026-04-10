"""
preprocess.py
Preprocesamiento de texto para entrenamiento de embeddings.

Carga el corpus bilingüe español-shiwilu y lo tokeniza para
entrenamiento de modelos de embeddings (FastText, Word2Vec).
"""

import re
from pathlib import Path

import pandas as pd

PUNCTUATION_PATTERN = re.compile(r'[!?.,;:¡¿"()\u201c\u201d]')
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")


def clean_text(text: str) -> str:
    """
    Limpia texto para tokenización.
    
    - Elimina signos de puntuación básicos
    - Colapsa espacios múltiples
    - Strip espacios al inicio/final
    """
    if not isinstance(text, str):
        return ""
    
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = text.strip()
    
    return text


def tokenize(text: str) -> list[str]:
    """Tokeniza texto por espacios."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def load_and_tokenize(
    filepath: str | Path,
    esp_col: str = "ESP_normalizado",
    shi_col: str = "SHIWILU_normalizado"
) -> list[list[str]]:
    """
    Carga corpus CSV y tokeniza ambos idiomas.
    
    Args:
        filepath: Ruta al archivo CSV
        esp_col: Nombre de columna para español
        shi_col: Nombre de columna para shiwilu
    
    Returns:
        Lista de oraciones tokenizadas (ambos idiomas concatenados)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    
    if esp_col not in df.columns:
        raise ValueError(f"Columna '{esp_col}' no encontrada en el CSV")
    if shi_col not in df.columns:
        raise ValueError(f"Columna '{shi_col}' no encontrada en el CSV")
    
    sentences: list[list[str]] = []
    
    for _, row in df.iterrows():
        esp_tokens = tokenize(str(row[esp_col]))
        if esp_tokens:
            sentences.append(esp_tokens)
        
        shi_tokens = tokenize(str(row[shi_col]))
        if shi_tokens:
            sentences.append(shi_tokens)
    
    return sentences
