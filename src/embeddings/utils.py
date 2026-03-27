"""
utils.py
Funciones utilitarias para trabajar con embeddings entrenados.

Provee acceso simple a vectores y búsqueda de palabras similares.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gensim.models import FastText


def get_vector(model: "FastText", word: str) -> np.ndarray:
    """
    Obtiene el vector de embedding para una palabra.
    
    FastText puede generar vectores para palabras OOV (fuera del vocabulario)
    usando información de subpalabras (n-gramas de caracteres).
    
    Args:
        model: Modelo FastText entrenado
        word: Palabra a consultar
    
    Returns:
        Vector numpy de dimensión igual a vector_size del modelo
    """
    return model.wv[word]


def get_similar_words(
    model: "FastText",
    word: str,
    topn: int = 10
) -> list[tuple[str, float]]:
    """
    Encuentra las palabras más similares a una palabra dada.
    
    Args:
        model: Modelo FastText entrenado
        word: Palabra de consulta
        topn: Número de palabras similares a retornar
    
    Returns:
        Lista de tuplas (palabra, similitud_coseno)
    """
    return model.wv.most_similar(word, topn=topn)


def word_in_vocab(model: "FastText", word: str) -> bool:
    """
    Verifica si una palabra está en el vocabulario del modelo.
    
    Nota: FastText puede generar vectores para palabras OOV,
    pero esta función indica si la palabra fue vista durante entrenamiento.
    
    Args:
        model: Modelo FastText entrenado
        word: Palabra a verificar
    
    Returns:
        True si la palabra está en el vocabulario
    """
    return word in model.wv.key_to_index


def get_vocab_size(model: "FastText") -> int:
    """Retorna el tamaño del vocabulario del modelo."""
    return len(model.wv.key_to_index)


def get_vector_size(model: "FastText") -> int:
    """Retorna la dimensión de los vectores del modelo."""
    return model.wv.vector_size
