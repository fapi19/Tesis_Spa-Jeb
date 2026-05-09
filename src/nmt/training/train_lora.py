"""Phase 4c: NLLB + LoRA bidirectional fine-tuning.

Wraps NLLB-200 distilled 600M with a single LoRA adapter targeting
q_proj/v_proj, trains on the bidirectional dataset built in Phase 4b, and
selects the best step by mean chrF++ across both directions (eval_avg_chrf).

Supports per-row loss weighting (Enhancement #4) when the dataset carries
``sample_weight`` features. Used to downweight synthetic pairs (mined and
backtranslated) relative to real parallel data in v1_bt training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

from .dataset import TokenizationConfig, load_filtered_splits
from .model_setup import (
    TokenizerExtensionConfig,
    build_extended_tokenizer,
    prepare_model_for_training,
)


@dataclass(frozen=True)
class LoraHyperparams:
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    task_type: str


@dataclass(frozen=True)
class TrainingHyperparams:
    output_dir: str
    seed: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    label_smoothing_factor: float
    max_source_length: int
    max_target_length: int
    precision: str
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    predict_with_generate: bool
    generation_num_beams: int
    generation_max_length: int
    logging_steps: int
    report_to: list[str]


@dataclass(frozen=True)
class DataPaths:
    train_csv: Path
    valid_csv: Path
    test_csv: Path
    source_column: str
    target_column: str
    source_lang_column: str
    target_lang_column: str
    lang_code_map: dict[str, str]


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    tokenizer_cfg: TokenizerExtensionConfig
    lora: LoraHyperparams
    training: TrainingHyperparams
    data: DataPaths

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path) -> "TrainingConfig":
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        lora_cfg = cfg["lora"]
        lora = LoraHyperparams(
            r=int(lora_cfg["r"]),
            alpha=int(lora_cfg["lora_alpha"]),
            dropout=float(lora_cfg["lora_dropout"]),
            bias=str(lora_cfg["bias"]),
            target_modules=tuple(lora_cfg["target_modules"]),
            task_type=str(lora_cfg["task_type"]),
        )

        tcfg = cfg["training"]
        training = TrainingHyperparams(
            output_dir=str(tcfg["output_dir"]),
            seed=int(tcfg.get("seed", 42)),
            learning_rate=float(tcfg["learning_rate"]),
            per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
            num_train_epochs=int(tcfg["num_train_epochs"]),
            warmup_ratio=float(tcfg["warmup_ratio"]),
            weight_decay=float(tcfg["weight_decay"]),
            lr_scheduler_type=str(tcfg["lr_scheduler_type"]),
            optim=str(tcfg["optim"]),
            label_smoothing_factor=float(tcfg.get("label_smoothing_factor", 0.0)),
            max_source_length=int(tcfg["max_source_length"]),
            max_target_length=int(tcfg["max_target_length"]),
            precision=str(tcfg.get("precision", "auto")),
            evaluation_strategy=str(tcfg["evaluation_strategy"]),
            eval_steps=int(tcfg["eval_steps"]),
            save_strategy=str(tcfg["save_strategy"]),
            save_steps=int(tcfg["save_steps"]),
            save_total_limit=int(tcfg.get("save_total_limit", 4)),
            load_best_model_at_end=bool(tcfg["load_best_model_at_end"]),
            metric_for_best_model=str(tcfg["metric_for_best_model"]),
            greater_is_better=bool(tcfg.get("greater_is_better", True)),
            predict_with_generate=bool(tcfg["predict_with_generate"]),
            generation_num_beams=int(tcfg["generation_num_beams"]),
            generation_max_length=int(tcfg["generation_max_length"]),
            logging_steps=int(tcfg.get("logging_steps", 25)),
            report_to=list(tcfg.get("report_to", [])),
        )

        d = cfg["data"]
        data = DataPaths(
            train_csv=project_root / d["train_csv"],
            valid_csv=project_root / d["valid_csv"],
            test_csv=project_root / d["test_csv"],
            source_column=str(d["source_column"]),
            target_column=str(d["target_column"]),
            source_lang_column=str(d["source_lang_column"]),
            target_lang_column=str(d["target_lang_column"]),
            lang_code_map={str(k): str(v) for k, v in d["lang_code_map"].items()},
        )

        return cls(
            base_model=str(cfg["base_model"]),
            tokenizer_cfg=TokenizerExtensionConfig(
                base_model=str(cfg["base_model"]),
                shiwilu_lang_code=str(cfg["tokenizer"]["shiwilu_lang_code"]),
                register_t5_style_tags=bool(cfg["tokenizer"].get("also_register_t5_style_tags", True)),
                init_neighbors=tuple(
                    cfg["tokenizer"].get("init_shw_from_neighbors", ["quy_Latn", "ayr_Latn", "grn_Latn"])
                ),
            ),
            lora=lora,
            training=training,
            data=data,
        )


def resolve_precision(requested: str) -> tuple[bool, bool]:
    """Return (fp16, bf16). Plan section 25 prescribes fp16 on CUDA."""
    if requested == "fp16":
        return True, False
    if requested == "bf16":
        return False, True
    if requested == "fp32":
        return False, False
    if requested == "auto":
        if torch.cuda.is_available():
            return True, False
        if torch.backends.mps.is_available():
            return False, True
        return False, False
    raise ValueError(f"unknown precision: {requested!r}")


def make_metrics_fn(tokenizer: PreTrainedTokenizerBase):
    chrf = sacrebleu.metrics.CHRF(word_order=2, char_order=6, beta=2)

    def _decode(ids: np.ndarray) -> list[str]:
        ids = np.where(ids != -100, ids, tokenizer.pad_token_id)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    def compute_metrics(eval_pred) -> dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = _decode(np.array(preds))
        decoded_labels = _decode(np.array(labels))
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrfpp = chrf.corpus_score(decoded_preds, [decoded_labels])
        return {
            "bleu": float(bleu.score),
            "chrf": float(chrfpp.score),
        }

    return compute_metrics


SAMPLE_WEIGHT_KEY = "sample_weight"


class Seq2SeqCollatorWithDecoderInputs:
    """Wrap `DataCollatorForSeq2Seq` and inject `decoder_input_ids`.

    transformers >=4.55 dropped `M2M100ForConditionalGeneration.prepare_decoder_input_ids_from_labels`,
    so the upstream collator silently leaves `decoder_input_ids` out of the batch.
    Combined with `label_smoothing_factor>0` (which makes `Trainer.compute_loss`
    pop `labels` out of `inputs` before the forward), the NLLB decoder ends up
    receiving neither `input_ids` nor `inputs_embeds` and raises the misleading
    `"You cannot specify both decoder_input_ids and decoder_inputs_embeds"` error.

    We replicate the previous behaviour by computing the shifted decoder ids
    here, using the model config we capture at construction time. This keeps us
    on the latest transformers/peft/accelerate releases.

    If features carry a ``sample_weight`` field (Enhancement #4), it is pulled
    out before the inner collator runs (it is not a tokenizer-aware feature)
    and re-attached to the batch as a ``[B]`` float tensor for downstream loss
    weighting in :class:`BidiSeq2SeqTrainer`.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        pad_token_id: int,
        decoder_start_token_id: int,
        label_pad_token_id: int = -100,
    ) -> None:
        self._inner = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=label_pad_token_id,
        )
        self._pad_token_id = int(pad_token_id)
        self._decoder_start_token_id = int(decoder_start_token_id)
        self._label_pad_token_id = int(label_pad_token_id)

    def __call__(self, features):
        weights: list[float] | None = None
        if features and SAMPLE_WEIGHT_KEY in features[0]:
            weights = [float(f.pop(SAMPLE_WEIGHT_KEY)) for f in features]
        batch = self._inner(features)
        labels = batch.get("labels", None)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels_for_shift = labels.masked_fill(
                    labels == self._label_pad_token_id, self._pad_token_id
                )
            else:
                labels_for_shift = torch.as_tensor(labels, dtype=torch.long)
                labels_for_shift = labels_for_shift.masked_fill(
                    labels_for_shift == self._label_pad_token_id, self._pad_token_id
                )
            batch["decoder_input_ids"] = shift_tokens_right(
                labels_for_shift,
                self._pad_token_id,
                self._decoder_start_token_id,
            )
        if weights is not None:
            batch[SAMPLE_WEIGHT_KEY] = torch.tensor(weights, dtype=torch.float32)
        return batch


def _weighted_smoothed_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    epsilon: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-row weighted CE with optional label smoothing.

    Replicates HF's :class:`LabelSmoother` formula at the per-token level, then
    scales each row by ``sample_weight[b]`` before reducing. The denominator
    is the *weighted* count of valid tokens, so a row with weight 0.3
    contributes 30% of its tokens to the average. With ``epsilon=0`` this
    collapses to plain weighted cross-entropy.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    nll = -log_probs.gather(
        dim=-1, index=labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # [B, T]
    smooth = -log_probs.mean(dim=-1)  # [B, T]
    pad_mask = labels.eq(ignore_index)  # [B, T]
    nll = nll.masked_fill(pad_mask, 0.0)
    smooth = smooth.masked_fill(pad_mask, 0.0)
    per_token = (1.0 - epsilon) * nll + epsilon * smooth  # [B, T]

    valid_per_row = (~pad_mask).sum(dim=-1).to(per_token.dtype)  # [B]
    weighted_token_loss = (sample_weight.to(per_token) * per_token.sum(dim=-1)).sum()
    weighted_token_count = (sample_weight.to(per_token) * valid_per_row).sum().clamp(min=1.0)
    return weighted_token_loss / weighted_token_count


class BidiSeq2SeqTrainer(Seq2SeqTrainer):
    """Inject `eval_avg_chrf` averaging chrF++ across both directions.

    When `eval_dataset` is a dict (as in this pipeline:
    {"shw2spa": ..., "spa2shw": ...}), HF emits per-direction metrics like
    `eval_shw2spa_chrf` and `eval_spa2shw_chrf`. We compute their mean and
    expose it under the prefixed key (default `eval_avg_chrf`) so
    `metric_for_best_model` can consume it.

    Also supports per-row loss weighting via the ``sample_weight`` batch
    field (Enhancement #4). When present, the trainer's HF
    ``label_smoother`` is bypassed and label smoothing is recomputed
    in-place at per-row granularity to keep the weighted average correct.
    """

    def __init__(self, *args, label_smoothing_factor: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._weighted_label_smoothing = float(label_smoothing_factor)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        if SAMPLE_WEIGHT_KEY not in inputs:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        sample_weight = inputs.pop(SAMPLE_WEIGHT_KEY)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = _weighted_smoothed_ce(
            outputs.logits,
            labels,
            sample_weight,
            epsilon=self._weighted_label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        chrf_keys = [
            k for k in metrics
            if k.startswith(f"{metric_key_prefix}_") and k.endswith("_chrf")
        ]
        directional = [k for k in chrf_keys if k != f"{metric_key_prefix}_chrf"]
        if directional:
            avg = float(np.mean([metrics[k] for k in directional]))
            metrics[f"{metric_key_prefix}_avg_chrf"] = avg
            self.log({f"{metric_key_prefix}_avg_chrf": avg})
        return metrics


def split_validation_by_direction(ds: Dataset) -> dict[str, Dataset]:
    out: dict[str, Dataset] = {}
    for direction in ("shw2spa", "spa2shw"):
        sub = ds.filter(lambda ex: ex["direction"] == direction)
        if len(sub) > 0:
            out[direction] = sub
    return out


def build_lora_model(
    cfg: TrainingConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    torch_dtype: torch.dtype | None = None,
):
    base_model = prepare_model_for_training(cfg.tokenizer_cfg, tokenizer, torch_dtype=torch_dtype)
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        task_type=cfg.lora.task_type,
    )
    return get_peft_model(base_model, lora_cfg)


def build_training_arguments(cfg: TrainingConfig) -> Seq2SeqTrainingArguments:
    fp16, bf16 = resolve_precision(cfg.training.precision)
    return Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        seed=cfg.training.seed,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        optim=cfg.training.optim,
        label_smoothing_factor=cfg.training.label_smoothing_factor,
        fp16=fp16,
        bf16=bf16,
        eval_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        predict_with_generate=cfg.training.predict_with_generate,
        generation_num_beams=cfg.training.generation_num_beams,
        generation_max_length=cfg.training.generation_max_length,
        logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to or [],
        dataloader_num_workers=2,            # Windows fork semantics
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        include_inputs_for_metrics=False,
        push_to_hub=False,
    )


def build_trainer(
    cfg: TrainingConfig,
    *,
    extra_train_csvs: Sequence[Path] | None = None,
    project_root: Path,
    weight_map: dict[str, float] | None = None,
    default_weight: float = 1.0,
) -> tuple[BidiSeq2SeqTrainer, dict[str, Any]]:
    """Wire model + datasets + trainer for one run.

    ``weight_map`` enables Enhancement #4: when provided (and the input CSVs
    carry ``origin_source``), each training row gets a ``sample_weight``
    feature drawn from the map. Synthetic origins typically map to <1.0 to
    downweight noisy pairs. Validation/test never get weights — they are
    dropped after tokenization.
    """
    tokenizer, tok_summary = build_extended_tokenizer(
        cfg.tokenizer_cfg,
        save_dir=project_root / "models" / "nmt" / "tokenizer_shw_extended",
    )

    fp16, _ = resolve_precision(cfg.training.precision)
    torch_dtype = torch.float32   # keep base weights fp32; fp16/bf16 enabled via TrainingArguments
    model = build_lora_model(cfg, tokenizer, torch_dtype=torch_dtype)
    model.print_trainable_parameters()

    tk_cfg = TokenizationConfig(
        max_source_length=cfg.training.max_source_length,
        max_target_length=cfg.training.max_target_length,
        lang_code_map=cfg.data.lang_code_map,
        weight_map=weight_map,
        default_weight=default_weight,
    )
    train_csvs: list[Path] = [cfg.data.train_csv]
    if extra_train_csvs:
        train_csvs.extend(extra_train_csvs)

    splits = load_filtered_splits(
        cfg.data.train_csv.parent,
        tokenizer,
        tk_cfg,
        train_csvs=train_csvs,
        valid_filename=cfg.data.valid_csv.name,
        test_filename=cfg.data.test_csv.name,
    )

    val_directional = split_validation_by_direction(splits["validation"])
    if not val_directional:
        raise RuntimeError("validation dataset has no rows for either direction")

    args = build_training_arguments(cfg)
    base_config = model.get_base_model().config if hasattr(model, "get_base_model") else model.config
    collator = Seq2SeqCollatorWithDecoderInputs(
        tokenizer,
        pad_token_id=base_config.pad_token_id,
        decoder_start_token_id=base_config.decoder_start_token_id,
    )

    use_weights = weight_map is not None
    train_columns = ["input_ids", "attention_mask", "labels"]
    if use_weights and SAMPLE_WEIGHT_KEY in splits["train"].column_names:
        train_columns.append(SAMPLE_WEIGHT_KEY)
    eval_columns = ["input_ids", "attention_mask", "labels"]

    train_for_loader = splits["train"].remove_columns(
        [c for c in splits["train"].column_names if c not in train_columns]
    )
    val_for_loader = {
        k: v.remove_columns([c for c in v.column_names if c not in eval_columns])
        for k, v in val_directional.items()
    }

    trainer_kwargs: dict[str, Any] = {}
    if use_weights:
        # When we own the loss, hand off label smoothing to our compute_loss
        # to keep the per-row weighting numerically equivalent to HF's
        # LabelSmoother on the unweighted path.
        trainer_kwargs["label_smoothing_factor"] = cfg.training.label_smoothing_factor
        object.__setattr__(args, "label_smoothing_factor", 0.0)

    trainer = BidiSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_for_loader,
        eval_dataset=val_for_loader,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=make_metrics_fn(tokenizer),
        **trainer_kwargs,
    )

    weight_summary: dict[str, Any] | None = None
    if use_weights and SAMPLE_WEIGHT_KEY in splits["train"].column_names:
        weights_arr = np.asarray(splits["train"][SAMPLE_WEIGHT_KEY], dtype=np.float64)
        weight_summary = {
            "map": dict(weight_map or {}),
            "default_weight": default_weight,
            "n_rows": int(weights_arr.size),
            "mean": float(weights_arr.mean()) if weights_arr.size else None,
            "min": float(weights_arr.min()) if weights_arr.size else None,
            "max": float(weights_arr.max()) if weights_arr.size else None,
            "label_smoothing_factor": cfg.training.label_smoothing_factor,
        }

    info = {
        "tokenizer_lang_ids": tok_summary,
        "train_rows": len(train_for_loader),
        "validation_rows_by_direction": {k: len(v) for k, v in val_for_loader.items()},
        "test_rows": len(splits["test"]),
        "lora_r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "fp16": fp16,
        "weighting": weight_summary,
    }
    return trainer, info
