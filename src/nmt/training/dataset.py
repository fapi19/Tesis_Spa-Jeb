"""Phase 4b: bidirectional dataset builder.

Reads the Phase 2 filtered CSVs, splits by direction (shw->spa and spa->shw),
tokenizes each direction with the appropriate src_lang/tgt_lang, then
concatenates back into one HuggingFace Dataset per split.

When a ``weight_map`` is provided in :class:`TokenizationConfig` and the CSV
has an ``origin_source`` column, every row receives a ``sample_weight`` field
that downstream training code can use to apply per-row loss weighting (used
to downweight synthetic pairs in v1_bt; Enhancement #4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase


DEFAULT_WEIGHT_MAP: dict[str, float] = {
    "flashcards": 1.0,
    "flashcards2": 1.0,
    "flashcards_oraciones": 1.0,
    "pdf_textos": 1.0,
    "fidel_lomas": 1.0,
    "vs_textos_narrativos": 1.0,
    "el_principito": 1.0,
    # Lexical sources (vocabulary-level, train-only): downweighted to 0.3 so
    # their distribution does not dominate the loss while their lexical signal
    # is still incorporated.
    "extra": 0.3,
    "cotidianas": 0.3,
    # Synthetic origins (augmentation):
    "mined_v3_sbert": 0.5,
    "backtranslation_v0": 0.3,
    "backtranslation_roundtrip_v0": 0.3,
    "morphological_variant": 0.3,
}


@dataclass(frozen=True)
class TokenizationConfig:
    max_source_length: int
    max_target_length: int
    lang_code_map: dict[str, str]   # plan codes (shw, spa) -> NLLB codes (shw_Latn, spa_Latn)
    weight_map: dict[str, float] | None = None
    default_weight: float = 1.0


def _read_filtered(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"id", "pair_id", "source", "target", "source_lang", "target_lang", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df = df.dropna(subset=["source", "target"]).reset_index(drop=True)
    return df


def _resolve_weights(
    df_block: pd.DataFrame,
    weight_map: dict[str, float] | None,
    default_weight: float,
) -> list[float] | None:
    """Map ``origin_source`` to a per-row weight when a map is provided.

    Returns ``None`` when no weighting should be applied (no map, or no
    ``origin_source`` column). Unknown ``origin_source`` values fall back to
    ``default_weight`` (1.0 by default) so adding a new source never silently
    drops to zero.
    """
    if not weight_map:
        return None
    if "origin_source" not in df_block.columns:
        return None
    sources = df_block["origin_source"].astype(str).tolist()
    return [float(weight_map.get(s, default_weight)) for s in sources]


def _tokenize_block(
    df_block: pd.DataFrame,
    src_plan: str,
    tgt_plan: str,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TokenizationConfig,
) -> Dataset:
    src_code = cfg.lang_code_map[src_plan]
    tgt_code = cfg.lang_code_map[tgt_plan]
    tokenizer.src_lang = src_code
    tokenizer.tgt_lang = tgt_code

    sources = df_block["source"].astype(str).tolist()
    targets = df_block["target"].astype(str).tolist()

    model_inputs = tokenizer(
        sources,
        max_length=cfg.max_source_length,
        truncation=True,
        padding=False,
    )
    # text_target sets tgt_lang prefix automatically in NLLB tokenizer
    target_inputs = tokenizer(
        text_target=targets,
        max_length=cfg.max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = target_inputs["input_ids"]

    record_payload = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"],
        "id": df_block["id"].astype(str).tolist(),
        "pair_id": df_block["pair_id"].astype(str).tolist(),
        "source_lang": [src_plan] * len(df_block),
        "target_lang": [tgt_plan] * len(df_block),
        "direction": [f"{src_plan}2{tgt_plan}"] * len(df_block),
    }
    weights = _resolve_weights(df_block, cfg.weight_map, cfg.default_weight)
    if weights is not None:
        record_payload["sample_weight"] = weights
    return Dataset.from_dict(record_payload)


def build_split_dataset(
    csv_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TokenizationConfig,
) -> Dataset:
    df = _read_filtered(csv_path)

    parts: list[Dataset] = []
    for src, tgt in (("shw", "spa"), ("spa", "shw")):
        block = df[(df["source_lang"] == src) & (df["target_lang"] == tgt)].copy()
        if len(block) == 0:
            continue
        parts.append(_tokenize_block(block, src, tgt, tokenizer, cfg))
    if not parts:
        raise RuntimeError(f"{csv_path}: no rows after direction split")
    ds = concatenate_datasets(parts)
    return ds


def load_filtered_splits(
    filtered_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TokenizationConfig,
    *,
    train_csvs: Iterable[Path] | None = None,
    valid_filename: str = "valid.csv",
    test_filename: str = "test.csv",
) -> dict[str, Dataset]:
    """Load and tokenize train/valid/test splits.

    train_csvs allows passing additional CSVs (e.g. for Phase 7d augmented
    training: filtered_dir/train.csv + 07_nmt_augmented/train_bt.csv + ...).
    They are concatenated row-wise before tokenization.
    """
    if train_csvs is None:
        train_csvs = [filtered_dir / "train.csv"]
    train_paths = list(train_csvs)
    if len(train_paths) == 1:
        train_ds = build_split_dataset(train_paths[0], tokenizer, cfg)
    else:
        # Concatenate the source CSVs, then tokenize once.
        merged = pd.concat(
            [_read_filtered(p) for p in train_paths],
            ignore_index=True,
        )
        merged_path = filtered_dir / "_train_merged_for_tokenize.tmp.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        try:
            train_ds = build_split_dataset(merged_path, tokenizer, cfg)
        finally:
            merged_path.unlink(missing_ok=True)

    valid_ds = build_split_dataset(filtered_dir / valid_filename, tokenizer, cfg)
    test_ds = build_split_dataset(filtered_dir / test_filename, tokenizer, cfg)

    return {"train": train_ds, "validation": valid_ds, "test": test_ds}
