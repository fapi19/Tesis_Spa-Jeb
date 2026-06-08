"""Phase 5 metrics: BLEU, chrF++, BERTScore, COMET.

chrF++ is the headline metric for low-resource morphology-rich languages
per plan section 31. BLEU is reported as secondary. BERTScore and COMET
are reported with explicit Shiwilu-OOD caveats.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sacrebleu
import yaml


@dataclass(frozen=True)
class MetricsConfig:
    primary_metric: str
    bleu_tokenize: str
    chrf_word_order: int
    chrf_char_order: int
    chrf_beta: int
    bertscore_model: str
    bertscore_num_layers: int
    bertscore_rescale_with_baseline: bool
    comet_model: str
    comet_batch_size: int

    @classmethod
    def from_yaml(cls, path: Path) -> "MetricsConfig":
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg["metrics"]
        return cls(
            primary_metric=str(cfg.get("primary_metric", "chrf_pp")),
            bleu_tokenize=str(m["bleu"].get("tokenize", "13a")),
            chrf_word_order=int(m["chrf_pp"].get("word_order", 2)),
            chrf_char_order=int(m["chrf_pp"].get("char_order", 6)),
            chrf_beta=int(m["chrf_pp"].get("beta", 2)),
            bertscore_model=str(m["bertscore"].get("model_type", "xlm-roberta-large")),
            bertscore_num_layers=int(m["bertscore"].get("num_layers", 17)),
            bertscore_rescale_with_baseline=bool(m["bertscore"].get("rescale_with_baseline", False)),
            comet_model=str(m["comet"].get("model", "Unbabel/wmt22-comet-da")),
            comet_batch_size=int(m["comet"].get("batch_size", 16)),
        )


def compute_bleu_chrf(hypotheses: list[str], references: list[str], cfg: MetricsConfig) -> dict[str, float]:
    bleu_metric = sacrebleu.metrics.BLEU(tokenize=cfg.bleu_tokenize)
    bleu_score = bleu_metric.corpus_score(hypotheses, [references])
    chrf_metric = sacrebleu.metrics.CHRF(
        word_order=cfg.chrf_word_order,
        char_order=cfg.chrf_char_order,
        beta=cfg.chrf_beta,
    )
    chrf_score = chrf_metric.corpus_score(hypotheses, [references])
    return {
        "bleu": float(bleu_score.score),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_pp": float(chrf_score.score),
        "chrf_pp_signature": str(chrf_metric.get_signature()),
    }


def sentence_bleu(hypothesis: str, reference: str, cfg: MetricsConfig) -> float:
    """Per-sentence BLEU (sacrebleu, same tokenization as :func:`compute_bleu_chrf`).

    Sentence-level BLEU is intentionally harsh on short hypotheses; the expert's
    "noise floor" reading (BLEU <= 10 is mostly frequent/short tokens) relies on
    this behaviour, so we keep the corpus tokenizer/smoothing and do not soften it.
    """
    metric = sacrebleu.metrics.BLEU(tokenize=cfg.bleu_tokenize, effective_order=True)
    return float(metric.sentence_score(hypothesis, [reference]).score)


def sentence_chrf_pp(hypothesis: str, reference: str, cfg: MetricsConfig) -> float:
    """Per-sentence chrF++ (character n-grams + word_order, same config as corpus)."""
    metric = sacrebleu.metrics.CHRF(
        word_order=cfg.chrf_word_order,
        char_order=cfg.chrf_char_order,
        beta=cfg.chrf_beta,
    )
    return float(metric.sentence_score(hypothesis, [reference]).score)


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    cfg: MetricsConfig,
    *,
    lang: str = "es",
) -> dict[str, float]:
    """BERTScore on xlm-roberta-large. Multilingual but with explicit Shiwilu OOD caveat."""
    import evaluate

    bertscore = evaluate.load("bertscore")
    result = bertscore.compute(
        predictions=hypotheses,
        references=references,
        model_type=cfg.bertscore_model,
        num_layers=cfg.bertscore_num_layers,
        rescale_with_baseline=cfg.bertscore_rescale_with_baseline,
        lang=lang,
        verbose=False,
    )
    return {
        "bertscore_p": float(sum(result["precision"]) / len(result["precision"])),
        "bertscore_r": float(sum(result["recall"]) / len(result["recall"])),
        "bertscore_f1": float(sum(result["f1"]) / len(result["f1"])),
        "bertscore_model": cfg.bertscore_model,
    }


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    cfg: MetricsConfig,
) -> dict[str, float]:
    """COMET (wmt22-comet-da). Reported as proxy on the Shiwilu side."""
    from comet import download_model, load_from_checkpoint

    model_path = download_model(cfg.comet_model)
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    output = model.predict(data, batch_size=cfg.comet_batch_size, gpus=0)
    system_score = float(output.system_score)
    return {
        "comet": system_score,
        "comet_model": cfg.comet_model,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    cfg: MetricsConfig,
    *,
    include_bertscore: bool = True,
    include_comet: bool = True,
) -> dict[str, Any]:
    """Compute the full metric suite, broken down per direction.

    `predictions` is a list of dicts with keys:
        id, direction, source, reference, hypothesis (top-1)
    """
    by_direction: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        by_direction.setdefault(p["direction"], []).append(p)

    out: dict[str, Any] = {
        "primary_metric": cfg.primary_metric,
        "directions": {},
        "caveats": {
            "bertscore": (
                "BERTScore uses xlm-roberta-large which has multilingual coverage but "
                "Shiwilu is OOD. Treat Shiwilu-side BERTScore as proxy only; "
                "prioritize chrF++ as the headline metric per plan section 31."
            ),
            "comet": (
                "COMET (wmt22-comet-da) was not trained on Shiwilu data. "
                "Reported as indicative only; not a reliable absolute number on the Shiwilu side."
            ),
        },
    }

    for direction, items in by_direction.items():
        sources = [p["source"] for p in items]
        hypotheses = [p["hypothesis"] for p in items]
        references = [p["reference"] for p in items]

        metrics: dict[str, Any] = {"n": len(items)}
        metrics.update(compute_bleu_chrf(hypotheses, references, cfg))

        if include_bertscore:
            lang = "es" if direction.endswith("2spa") else "es"   # use 'es' anchor; xlm-r-large is multilingual
            metrics.update(compute_bertscore(hypotheses, references, cfg, lang=lang))

        if include_comet:
            metrics.update(compute_comet(sources, hypotheses, references, cfg))

        out["directions"][direction] = metrics

    if all(d in out["directions"] for d in ("shw2spa", "spa2shw")):
        chrfs = [out["directions"][d]["chrf_pp"] for d in ("shw2spa", "spa2shw")]
        bleus = [out["directions"][d]["bleu"] for d in ("shw2spa", "spa2shw")]
        out["avg_chrf_pp"] = sum(chrfs) / len(chrfs)
        out["avg_bleu"] = sum(bleus) / len(bleus)

    return out


def prewarm_models(cfg: MetricsConfig) -> None:
    """Pre-download BERTScore + COMET checkpoints (Phase 5 prep on Windows
    avoids mid-run download timeouts).
    """
    print("[metrics] pre-warming BERTScore (xlm-roberta-large) ...")
    import evaluate
    bs = evaluate.load("bertscore")
    bs.compute(
        predictions=["hola"],
        references=["hola"],
        model_type=cfg.bertscore_model,
        num_layers=cfg.bertscore_num_layers,
        lang="es",
        verbose=False,
    )
    print("[metrics] pre-warming COMET ...")
    from comet import download_model, load_from_checkpoint
    path = download_model(cfg.comet_model)
    load_from_checkpoint(path)
    print("[metrics] pre-warm complete")
