from __future__ import annotations

import argparse
import torch
import sentencepiece as spm

from .model import EmbeddingEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    model = EmbeddingEncoder(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
    )
    model.load_state_dict(ckpt["model_state_dict"])

    matrix = model.embedding.weight.detach().cpu()
    torch.save(matrix, args.output)

    print(f"Matriz guardada en {args.output} con shape {tuple(matrix.shape)}")


if __name__ == "__main__":
    main()