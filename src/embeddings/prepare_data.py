from __future__ import annotations

from .preprocess_embeddings import normalize_text as canonical_normalize_text
from .preprocess_embeddings import preprocess_embeddings


def normalize_text(text: str) -> str:
    return canonical_normalize_text(text, language="shiwilu")


def main() -> None:
    splits = preprocess_embeddings()
    print("prepare_data ahora delega al pipeline canónico de embeddings.")
    print(f"Train: {len(splits['train'])}")
    print(f"Valid: {len(splits['valid'])}")
    print(f"Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()