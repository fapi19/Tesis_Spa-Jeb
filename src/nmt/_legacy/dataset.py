from __future__ import annotations

import json
from pathlib import Path
from typing import List

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class ParallelNMTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        sp_model: str,
        src_key: str,
        tgt_key: str,
        max_len: int = 160,
    ):
        self.rows = []
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.max_len = max_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def encode(self, text: str) -> List[int]:
        ids = self.sp.encode(text, out_type=int)[: self.max_len - 2]
        return [self.sp.bos_id(), *ids, self.sp.eos_id()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        src = self.encode(row[self.src_key])
        tgt = self.encode(row[self.tgt_key])

        return {
            "src_ids": src,
            "tgt_ids": tgt,
            "src_text": row[self.src_key],
            "tgt_text": row[self.tgt_key],
        }


def collate_nmt(batch, pad_id: int):
    def pad(seqs):
        max_len = max(len(x) for x in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return out

    src_ids = pad([x["src_ids"] for x in batch])
    tgt_ids = pad([x["tgt_ids"] for x in batch])

    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "src_text": [x["src_text"] for x in batch],
        "tgt_text": [x["tgt_text"] for x in batch],
    }