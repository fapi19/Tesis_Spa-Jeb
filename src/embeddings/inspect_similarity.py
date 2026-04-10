from __future__ import annotations

import argparse
import torch
import sentencepiece as spm

from .model import EmbeddingEncoder


def encode_text(sp, text: str, max_len: int = 64):
    ids = sp.encode(text, out_type=int)[: max_len - 2]
    return torch.tensor([[sp.bos_id(), *ids, sp.eos_id()]], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate_file", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    model = EmbeddingEncoder(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    q_ids = encode_text(sp, args.query)
    with torch.no_grad():
        q_emb = model.forward_sentence(q_ids)

    candidates = []
    with open(args.candidate_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                candidates.append(text)

    scored = []
    with torch.no_grad():
        for text in candidates:
            ids = encode_text(sp, text)
            emb = model.forward_sentence(ids)
            score = torch.cosine_similarity(q_emb, emb).item()
            scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    for text, score in scored[:10]:
        print(f"{score:.4f}\t{text}")


if __name__ == "__main__":
    main()