from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        pad_id: int = 0,
        share_embeddings: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        if share_embeddings:
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        self.pos_enc = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        if not torch.cuda.is_available():
            self.transformer.encoder.enable_nested_tensor = False
            self.transformer.encoder.use_nested_tensor = False

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, vocab_size)

        if share_embeddings:
            self.output_proj.weight = self.tgt_embed.weight

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        src_key_padding_mask = src_ids.eq(self.pad_id)
        tgt_key_padding_mask = tgt_input_ids.eq(self.pad_id)
        tgt_mask = self._generate_square_subsequent_mask(tgt_input_ids.size(1), tgt_input_ids.device)

        src = self.src_embed(src_ids) * math.sqrt(self.d_model)
        tgt = self.tgt_embed(tgt_input_ids) * math.sqrt(self.d_model)

        src = self.dropout(self.pos_enc(src))
        tgt = self.dropout(self.pos_enc(tgt))

        hidden = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(hidden)