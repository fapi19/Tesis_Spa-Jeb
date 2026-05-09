"""Phase 4a: extend the NLLB tokenizer + model for Shiwilu (shw_Latn).

NLLB-200 has no Shiwilu code. Per plan.md sections 18-21 we register
shw_Latn as a new FLORES-style language token (so it integrates with the
forced_bos_token_id API used during generation), keep the plan's t5-style
<2shw>/<2spa> tags as compatibility additional special tokens, and
mean-initialize the new shw_Latn embedding from three Andean / South-American
Indigenous neighbors (quy_Latn, ayr_Latn, grn_Latn).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
try:
    from transformers.models.nllb.tokenization_nllb_fast import NllbTokenizerFast
except Exception:  # pragma: no cover
    NllbTokenizerFast = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TokenizerExtensionConfig:
    base_model: str
    shiwilu_lang_code: str
    register_t5_style_tags: bool
    init_neighbors: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "TokenizerExtensionConfig":
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tok = cfg["tokenizer"]
        return cls(
            base_model=cfg["base_model"],
            shiwilu_lang_code=str(tok["shiwilu_lang_code"]),
            register_t5_style_tags=bool(tok.get("also_register_t5_style_tags", True)),
            init_neighbors=tuple(tok.get("init_shw_from_neighbors", ["quy_Latn", "ayr_Latn", "grn_Latn"])),
        )


def _additional_special_tokens(cfg: TokenizerExtensionConfig) -> list[str]:
    extra = [cfg.shiwilu_lang_code]
    if cfg.register_t5_style_tags:
        extra += ["<2shw>", "<2spa>"]
    return extra


def _register_lang_code_in_tokenizer(tokenizer: PreTrainedTokenizerBase, lang_code: str) -> int:
    """Update NLLB's lang_code_to_id / id_to_lang_code maps for the new code.

    NllbTokenizer (slow) builds these maps lazily from the special tokens
    registry; after add_special_tokens we assign the new token's id manually
    so downstream APIs (forced_bos_token_id, set_src_lang_special_tokens)
    keep working. NllbTokenizerFast keeps an analogous mapping.
    """
    new_id = tokenizer.convert_tokens_to_ids(lang_code)
    if new_id is None or new_id == tokenizer.unk_token_id:
        raise RuntimeError(f"failed to register {lang_code!r}: token id resolves to UNK")
    if isinstance(tokenizer, NllbTokenizer) or (
        NllbTokenizerFast is not None and isinstance(tokenizer, NllbTokenizerFast)
    ):
        if not hasattr(tokenizer, "lang_code_to_id"):
            tokenizer.lang_code_to_id = {}
        if not hasattr(tokenizer, "id_to_lang_code"):
            tokenizer.id_to_lang_code = {}
        tokenizer.lang_code_to_id[lang_code] = new_id  # type: ignore[index]
        tokenizer.id_to_lang_code[new_id] = lang_code  # type: ignore[index]
        if hasattr(tokenizer, "fairseq_tokens_to_ids"):
            tokenizer.fairseq_tokens_to_ids[lang_code] = new_id  # type: ignore[index]
        if hasattr(tokenizer, "fairseq_ids_to_tokens"):
            tokenizer.fairseq_ids_to_tokens[new_id] = lang_code  # type: ignore[index]
    return new_id


def build_extended_tokenizer(
    cfg: TokenizerExtensionConfig,
    save_dir: Path | None = None,
) -> tuple[PreTrainedTokenizerBase, dict[str, int]]:
    """Return (tokenizer, lang_code_to_id_summary)."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    extra = _additional_special_tokens(cfg)
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add = [t for t in extra if t not in existing]
    n_added = 0
    if to_add:
        n_added = tokenizer.add_special_tokens({"additional_special_tokens": list(existing) + to_add})

    new_id = _register_lang_code_in_tokenizer(tokenizer, cfg.shiwilu_lang_code)

    summary: dict[str, int] = {cfg.shiwilu_lang_code: int(new_id)}
    for code in cfg.init_neighbors:
        rid = tokenizer.convert_tokens_to_ids(code)
        if rid is None or rid == tokenizer.unk_token_id:
            raise RuntimeError(f"reference lang {code!r} not found in NLLB tokenizer")
        summary[code] = int(rid)
    spa_id = tokenizer.convert_tokens_to_ids("spa_Latn")
    if spa_id is None or spa_id == tokenizer.unk_token_id:
        raise RuntimeError("spa_Latn missing from NLLB tokenizer (sanity check failed)")
    summary["spa_Latn"] = int(spa_id)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_dir))

    return tokenizer, summary


def mean_init_new_embedding(
    model: AutoModelForSeq2SeqLM,
    new_token_id: int,
    neighbor_ids: Iterable[int],
) -> None:
    """Overwrite the new token's row in the input embeddings with the mean of
    its neighbors. NLLB-distilled has tied embeddings, so the lm_head row is
    updated implicitly.
    """
    embed_layer = model.get_input_embeddings()
    weight = embed_layer.weight.data
    neighbor_ids = list(neighbor_ids)
    if not neighbor_ids:
        raise ValueError("no neighbor ids provided for mean-init")
    neighbor_rows = weight[neighbor_ids].to(torch.float32)
    mean_vec = neighbor_rows.mean(dim=0).to(weight.dtype)
    with torch.no_grad():
        weight[new_token_id].copy_(mean_vec)
        if not getattr(model.config, "tie_word_embeddings", False):
            output_embed = model.get_output_embeddings()
            if output_embed is not None and output_embed.weight.shape == weight.shape:
                output_embed.weight.data[new_token_id].copy_(mean_vec)


def prepare_model_for_training(
    cfg: TokenizerExtensionConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    torch_dtype: torch.dtype | None = None,
) -> AutoModelForSeq2SeqLM:
    """Load NLLB, resize embeddings to the extended tokenizer, mean-init."""
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch_dtype,
    )
    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(cfg.shiwilu_lang_code)
    neighbor_ids = [tokenizer.convert_tokens_to_ids(code) for code in cfg.init_neighbors]
    mean_init_new_embedding(model, int(new_id), neighbor_ids)
    return model


def lang_code_for(plan_lang: str, lang_code_map: dict[str, str]) -> str:
    if plan_lang not in lang_code_map:
        raise KeyError(f"unknown plan language code {plan_lang!r}; known: {list(lang_code_map)}")
    return lang_code_map[plan_lang]
