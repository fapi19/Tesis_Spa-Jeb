from __future__ import annotations

import argparse
import json
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


def load_jsonl_pairs(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sp_model", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--show_n", type=int, default=10, help="cuántos ejemplos mostrar")
    parser.add_argument(
        "--direction",
        choices=["shw2es", "es2shw"],
        default="shw2es",
        help="dirección del retrieval: query shiwilu a corpus español, o al revés",
    )
    parser.add_argument(
        "--sort_by",
        choices=["original", "best", "worst"],
        default="original",
        help="cómo ordenar los ejemplos mostrados",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, sp = load_model(args.checkpoint, args.sp_model, device)

    rows = load_jsonl_pairs(args.val_jsonl)
    shw_texts = [r["shiwilu"] for r in rows]
    es_texts = [r["spanish"] for r in rows]

    shw_embs = embed_texts(model, sp, shw_texts, device)
    es_embs = embed_texts(model, sp, es_texts, device)

    if args.direction == "shw2es":
        query_texts = shw_texts
        corpus_texts = es_texts
        sims = shw_embs @ es_embs.T
        query_label = "Shiwilu query"
        correct_label = "Correcto español"
    else:
        query_texts = es_texts
        corpus_texts = shw_texts
        sims = es_embs @ shw_embs.T
        query_label = "Español query"
        correct_label = "Correcto shiwilu"

    per_example = []
    for i in range(len(rows)):
        row_scores = sims[i]
        topk = torch.topk(row_scores, k=min(args.top_k, len(corpus_texts)))
        top_indices = topk.indices.tolist()
        top_scores = topk.values.tolist()

        ranking = torch.argsort(row_scores, descending=True)
        correct_rank = (ranking == i).nonzero(as_tuple=True)[0].item() + 1
        correct_score = float(row_scores[i].item())
        top1_score = float(top_scores[0])
        score_gap = top1_score - correct_score

        per_example.append(
            {
                "i": i,
                "query": query_texts[i],
                "correct": corpus_texts[i],
                "correct_rank": correct_rank,
                "correct_score": correct_score,
                "top_indices": top_indices,
                "top_scores": top_scores,
                "top1_score": top1_score,
                "score_gap": score_gap,
            }
        )

    if args.sort_by == "best":
        per_example.sort(key=lambda x: x["correct_rank"])
    elif args.sort_by == "worst":
        per_example.sort(key=lambda x: x["correct_rank"], reverse=True)

    total = min(args.show_n, len(per_example))
    hits_at_1 = sum(1 for ex in per_example if ex["correct_rank"] == 1)
    hits_at_5 = sum(1 for ex in per_example if ex["correct_rank"] <= 5)
    hits_at_10 = sum(1 for ex in per_example if ex["correct_rank"] <= 10)
    median_rank = sorted(ex["correct_rank"] for ex in per_example)[len(per_example) // 2]

    for shown_idx, ex in enumerate(per_example[:total]):
        print("=" * 80)
        print(f"Ejemplo mostrado {shown_idx} | índice original {ex['i']}")
        print(f"{query_label}:      {ex['query']}")
        print(f"{correct_label}:   {ex['correct']}")
        print(f"Rank correcto:      {ex['correct_rank']}")
        print(f"Score correcto:     {ex['correct_score']:.4f}")
        print(f"Score top-1:        {ex['top1_score']:.4f}")
        print(f"Gap top1-correcto:  {ex['score_gap']:.4f}")
        print("\nTop candidatos:")
        for rank, (idx, score) in enumerate(zip(ex["top_indices"], ex["top_scores"]), start=1):
            mark = "  <-- CORRECTO" if idx == ex["i"] else ""
            print(f"{rank:02d}. score={score:.4f} | {corpus_texts[idx]}{mark}")

    print("\n" + "=" * 80)
    print(f"Resumen global sobre {len(per_example)} ejemplos")
    print(f"Hits@1 globales:  {hits_at_1}/{len(per_example)} = {hits_at_1/len(per_example):.4f}")
    print(f"Hits@5 globales:  {hits_at_5}/{len(per_example)} = {hits_at_5/len(per_example):.4f}")
    print(f"Hits@10 globales: {hits_at_10}/{len(per_example)} = {hits_at_10/len(per_example):.4f}")
    print(f"Mediana del rank correcto: {median_rank}")
    print(f"\nResumen sobre {total} ejemplos mostrados")


if __name__ == "__main__":
    main()