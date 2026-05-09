"""Phase 5 inference: beam-search generation for NLLB + LoRA checkpoints.

Returns both top-1 hypotheses (with their sequence_score) and the full top-K
n-best list (used by the Phase 6 semantic reranker).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

_FLORES_LANG_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def _ensure_extended_lang_codes_registered(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Re-register FLORES-style codes from `additional_special_tokens`.

    `tokenizer.lang_code_to_id` / `id_to_lang_code` are rebuilt by
    `from_pretrained` from NLLB-200's hardcoded language list. Custom codes we
    added during training (e.g. `shw_Latn`) survive in
    `special_tokens_map.json` but vanish from those dicts on reload, breaking
    `forced_bos_token_id` and `set_src_lang_special_tokens`. We restore them
    here without depending on the training config.
    """
    extras = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    if not hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.lang_code_to_id = {}  # type: ignore[attr-defined]
    if not hasattr(tokenizer, "id_to_lang_code"):
        tokenizer.id_to_lang_code = {}  # type: ignore[attr-defined]
    registered: list[str] = []
    for token in extras:
        if not _FLORES_LANG_CODE_RE.match(token):
            continue
        if token in tokenizer.lang_code_to_id:  # type: ignore[operator]
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            continue
        tokenizer.lang_code_to_id[token] = token_id  # type: ignore[index]
        tokenizer.id_to_lang_code[token_id] = token  # type: ignore[index]
        if hasattr(tokenizer, "fairseq_tokens_to_ids"):
            tokenizer.fairseq_tokens_to_ids[token] = token_id  # type: ignore[index]
        if hasattr(tokenizer, "fairseq_ids_to_tokens"):
            tokenizer.fairseq_ids_to_tokens[token_id] = token  # type: ignore[index]
        registered.append(token)
    return registered


@dataclass(frozen=True)
class GenerationConfig:
    num_beams: int
    length_penalty: float
    max_new_tokens: int
    no_repeat_ngram_size: int
    early_stopping: bool
    num_return_sequences: int
    per_device_batch_size: int

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerationConfig":
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gen = cfg["generation"]
        topk = cfg["topk"]
        batch = cfg.get("batch", {"per_device_batch_size": 8})
        return cls(
            num_beams=int(gen["num_beams"]),
            length_penalty=float(gen["length_penalty"]),
            max_new_tokens=int(gen["max_new_tokens"]),
            no_repeat_ngram_size=int(gen.get("no_repeat_ngram_size", 0)),
            early_stopping=bool(gen.get("early_stopping", True)),
            num_return_sequences=int(topk["num_return_sequences"]),
            per_device_batch_size=int(batch["per_device_batch_size"]),
        )


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    base_model: str,
    device: str = "auto",
    torch_dtype: torch.dtype | None = None,
) -> tuple[PeftModel, PreTrainedTokenizerBase, str]:
    """Load NLLB base + LoRA adapter from `checkpoint_dir`.

    The checkpoint dir must contain (a) the LoRA adapter (adapter_config.json
    + adapter_model.safetensors) and (b) the extended tokenizer.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    restored = _ensure_extended_lang_codes_registered(tokenizer)
    if restored:
        print(f"[inference] re-registered FLORES codes from checkpoint: {restored}")

    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model.eval()
    model.to(device)
    return model, tokenizer, device


def _resolve_lang_code(tokenizer: PreTrainedTokenizerBase, plan_lang: str, lang_code_map: dict[str, str]) -> str:
    if plan_lang not in lang_code_map:
        raise KeyError(f"unknown plan lang {plan_lang!r}; map keys: {list(lang_code_map)}")
    code = lang_code_map[plan_lang]
    if not hasattr(tokenizer, "lang_code_to_id") or code not in tokenizer.lang_code_to_id:
        raise RuntimeError(f"{code!r} not registered in tokenizer.lang_code_to_id")
    return code


@torch.no_grad()
def generate_for_direction(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: pd.DataFrame,
    *,
    src_plan: str,
    tgt_plan: str,
    lang_code_map: dict[str, str],
    cfg: GenerationConfig,
    device: str,
    return_topk: bool = True,
) -> list[dict[str, Any]]:
    src_code = _resolve_lang_code(tokenizer, src_plan, lang_code_map)
    tgt_code = _resolve_lang_code(tokenizer, tgt_plan, lang_code_map)
    forced_bos = tokenizer.lang_code_to_id[tgt_code]

    tokenizer.src_lang = src_code
    sources = rows["source"].astype(str).tolist()
    references = rows["target"].astype(str).tolist()
    ids = rows["id"].astype(str).tolist()
    pair_ids = rows["pair_id"].astype(str).tolist()

    batch_size = cfg.per_device_batch_size
    out: list[dict[str, Any]] = []
    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start : start + batch_size]
        batch_refs = references[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        batch_pids = pair_ids[start : start + batch_size]

        enc = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        gen_kwargs = dict(
            **enc,
            forced_bos_token_id=forced_bos,
            num_beams=cfg.num_beams,
            length_penalty=cfg.length_penalty,
            max_new_tokens=cfg.max_new_tokens,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            early_stopping=cfg.early_stopping,
            output_scores=True,
            return_dict_in_generate=True,
        )

        if return_topk:
            gen_kwargs["num_return_sequences"] = cfg.num_return_sequences

        gen_out = model.generate(**gen_kwargs)
        seq = gen_out.sequences
        scores = gen_out.sequences_scores if hasattr(gen_out, "sequences_scores") else None
        if scores is None:
            scores = torch.zeros(seq.shape[0], device=seq.device)

        decoded = tokenizer.batch_decode(seq, skip_special_tokens=True)

        n_per_input = cfg.num_return_sequences if return_topk else 1
        for i in range(len(batch_sources)):
            cand_start = i * n_per_input
            cand_end = cand_start + n_per_input
            candidates = []
            for j, h in enumerate(decoded[cand_start:cand_end]):
                candidates.append(
                    {
                        "hypothesis": h.strip(),
                        "sequence_score": float(scores[cand_start + j].item()),
                        "rank": j,
                    }
                )
            out.append(
                {
                    "id": batch_ids[i],
                    "pair_id": batch_pids[i],
                    "source_lang": src_plan,
                    "target_lang": tgt_plan,
                    "direction": f"{src_plan}2{tgt_plan}",
                    "source": batch_sources[i],
                    "reference": batch_refs[i],
                    "hypothesis": candidates[0]["hypothesis"],
                    "sequence_score": candidates[0]["sequence_score"],
                    "candidates": candidates,
                }
            )
    return out


def predict_split(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    csv_path: Path,
    *,
    cfg: GenerationConfig,
    lang_code_map: dict[str, str],
    device: str,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.dropna(subset=["source", "target"]).reset_index(drop=True)

    out: list[dict[str, Any]] = []
    for src_plan, tgt_plan in (("shw", "spa"), ("spa", "shw")):
        sub = df[(df["source_lang"] == src_plan) & (df["target_lang"] == tgt_plan)].copy()
        if len(sub) == 0:
            continue
        out.extend(
            generate_for_direction(
                model,
                tokenizer,
                sub,
                src_plan=src_plan,
                tgt_plan=tgt_plan,
                lang_code_map=lang_code_map,
                cfg=cfg,
                device=device,
                return_topk=True,
            )
        )
    return out
