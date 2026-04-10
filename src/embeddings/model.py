from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 768,
        pad_id: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=torch.cuda.is_available(),
        )

        self.final_ln = nn.LayerNorm(d_model)
        self.sentence_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward_tokens(self, input_ids: torch.Tensor):
        mask = input_ids.eq(self.pad_id)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embedding_dropout(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.final_ln(x)
        return x, mask

    def forward_sentence(self, input_ids: torch.Tensor):
        x, mask = self.forward_tokens(input_ids)
        lengths = (~mask).sum(dim=1).unsqueeze(1).clamp(min=1)
        sent = x.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / lengths
        sent = self.sentence_dropout(sent)
        sent = F.normalize(self.proj(sent), dim=-1)
        return sent