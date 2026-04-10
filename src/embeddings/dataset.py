from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class ParallelEmbeddingDataset(Dataset):
    def __init__(
        self,
        path: str,
        sp_model: str,
        max_len: int = 64,
        subword_regularization: bool = False,
        nbest_size: int = -1,
        alpha: float = 0.1,
    ):
        self.rows: List[Dict[str, str]] = []
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)
        self.max_len = max_len
        self.subword_regularization = subword_regularization
        self.nbest_size = nbest_size
        self.alpha = alpha

        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def encode(self, text: str) -> List[int]:
        if self.subword_regularization:
            ids = self.sp.encode(
                text,
                out_type=int,
                enable_sampling=True,
                nbest_size=self.nbest_size,
                alpha=self.alpha,
            )
        else:
            ids = self.sp.encode(text, out_type=int)

        ids = ids[: self.max_len - 2]
        return [self.sp.bos_id()] + ids + [self.sp.eos_id()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return {
            "shw_ids": self.encode(row["shiwilu"]),
            "es_ids": self.encode(row["spanish"]),
            "shw_text": row["shiwilu"],
            "es_text": row["spanish"],
        }


def collate_batch(batch, pad_id: int):
    def pad(seqs):
        max_len = max(len(x) for x in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    return {
        "shw_ids": pad([x["shw_ids"] for x in batch]),
        "es_ids": pad([x["es_ids"] for x in batch]),
        "shw_text": [x["shw_text"] for x in batch],
        "es_text": [x["es_text"] for x in batch],
    }