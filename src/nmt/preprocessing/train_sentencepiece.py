"""Phase 3: SentencePiece Unigram (analytic artifact).

Per plan.md sections 15-17, this SP model is **not** the runtime tokenizer
(NLLB ships its own). Purpose:
    1. Vocabulary / morphology analysis to defend the agglutinative-friendly choice.
    2. Comparison fixture against NLLB's tokenization (token-count and
       segmentation deltas on Shiwilu sentences).

Input: accepted-train Shiwilu + Spanish concatenated, one sentence per line.
Output: models/nmt/sentencepiece/sp_unigram_8k.{model,vocab}.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import sentencepiece as spm


@dataclass(frozen=True)
class SentencePieceConfig:
    vocab_size: int = 8000
    char_coverage: float = 1.0
    model_type: str = "unigram"
    model_prefix: str = "sp_unigram_8k"


def collect_corpus(filtered_train_csv: Path) -> list[str]:
    """Return the list of accepted-train sentences (Shiwilu + Spanish)."""
    df = pd.read_csv(filtered_train_csv, encoding="utf-8-sig")
    shw = df[df["source_lang"] == "shw"][["pair_id", "source", "target"]]
    shw = shw.drop_duplicates(subset=["pair_id"])

    sentences: list[str] = []
    for _, row in shw.iterrows():
        if isinstance(row["source"], str) and row["source"].strip():
            sentences.append(row["source"].strip())
        if isinstance(row["target"], str) and row["target"].strip():
            sentences.append(row["target"].strip())
    return sentences


def write_corpus(sentences: Iterable[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.replace("\n", " ") + "\n")
            written += 1
    return written


def train_sentencepiece(
    corpus_path: Path,
    out_dir: Path,
    cfg: SentencePieceConfig,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / cfg.model_prefix
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=cfg.vocab_size,
        character_coverage=cfg.char_coverage,
        model_type=cfg.model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        normalization_rule_name="identity",  # preserve apostrophes / morphology
        input_sentence_size=0,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
        treat_whitespace_as_suffix=False,
        # ~5k accepted-train sentences make the requested 8000-vocab unreachable
        # (the corpus has roughly 4-5k distinct subword candidates). Treat
        # vocab_size as an upper bound and report the actual size in
        # sentencepiece_stats.json.
        hard_vocab_limit=False,
    )
    return prefix.with_suffix(".model"), prefix.with_suffix(".vocab")
