from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from .model import EmbeddingEncoder


def encode_text(sp: spm.SentencePieceProcessor, text: str, max_len: int = 64) -> torch.Tensor:
    ids = sp.encode(text, out_type=int)[: max_len - 2]
    ids = [sp.bos_id(), *ids, sp.eos_id()]
    return torch.tensor([ids], dtype=torch.long)


def load_model(checkpoint_path: str, sp_model_path: str, device: torch.device) -> tuple[EmbeddingEncoder, spm.SentencePieceProcessor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    model = EmbeddingEncoder(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        nhead=ckpt["nhead"],
        num_layers=ckpt["num_layers"],
        ff_dim=ckpt["ff_dim"],
        pad_id=ckpt["pad_id"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, sp


@torch.no_grad()
def embed_texts(
    model: EmbeddingEncoder,
    sp: spm.SentencePieceProcessor,
    texts: list[str],
    device: torch.device,
    max_len: int = 64,
) -> torch.Tensor:
    embs = []
    for text in texts:
        ids = encode_text(sp, text, max_len=max_len).to(device)
        emb = model.forward_sentence(ids).cpu()
        embs.append(emb)
    return torch.cat(embs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--corpus_file", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, sp = load_model(args.checkpoint, args.sp_model, device)

    corpus = [line.strip() for line in Path(args.corpus_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_embs = embed_texts(model, sp, corpus, device)

    while True:
        query = input("\nEscribe una frase (o ENTER para salir): ").strip()
        if not query:
            break

        query_emb = embed_texts(model, sp, [query], device)[0]
        sims = corpus_embs @ query_emb
        topk = torch.topk(sims, k=min(args.top_k, len(corpus)))

        print("\nMás parecidas:")
        for rank, (idx, score) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
            print(f"{rank:02d}. score={score:.4f} | {corpus[idx]}")


if __name__ == "__main__":
    main()