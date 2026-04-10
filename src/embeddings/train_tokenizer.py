from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def train_tokenizer(
    input_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 4000,
    model_type: str = "unigram",
) -> Path:
    """Train a SentencePiece tokenizer and return the .model path."""
    model_prefix = str(model_prefix)
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    out = Path(f"{model_prefix}.model")
    print(f"Tokenizer guardado en {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--model_type", choices=["unigram", "bpe"], default="unigram")
    args = parser.parse_args()

    train_tokenizer(args.input, args.model_prefix, args.vocab_size, args.model_type)


if __name__ == "__main__":
    main()
