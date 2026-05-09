"""Phase 7a: backtranslation augmentation.

Pipeline:
    1. Build a monolingual Shiwilu pool from candidate text sources, dropping
       lines that are already in the parallel corpus or that look Spanish.
    2. Translate the mono pool with a frozen NLLB+LoRA checkpoint
       (shw -> spa direction) using beam=5.
    3. Filter the synthetic pairs through the Phase 2 semantic filter
       (only keep pairs with cos_sim > flag_upper, i.e. accepted).
    4. Cap the resulting set to <= bt_cap_x_parallel * |accepted_train|.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Common Spanish stopwords for the heuristic filter. Whole-token match.
_SPANISH_STOPWORDS = frozenset(
    [
        "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
        "de", "del", "al", "a", "en", "por", "para", "con", "sin",
        "sobre", "entre", "como", "que", "qué", "porque", "porqué",
        "y", "e", "ni", "o", "u", "pero", "sino",
        "no", "sí", "ya", "muy", "más", "menos", "mucho", "poco",
        "es", "está", "estoy", "son", "están", "fue", "fueron", "ha",
        "han", "hay", "tiene", "tienen", "tengo", "vamos", "voy", "va",
        "se", "le", "les", "me", "mi", "mis", "tu", "tus", "su", "sus",
        "yo", "tú", "él", "ella", "nosotros", "ustedes", "ellos", "ellas",
        "este", "esta", "esto", "estos", "estas", "ese", "esa", "eso", "esos", "esas",
        "ser", "estar", "tener", "hacer", "poder",
    ]
)
_TOKEN_RE = re.compile(r"[A-Za-záéíóúñü']+", flags=re.IGNORECASE)


def _normalize_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _looks_spanish(line: str, stopword_threshold: int = 1) -> bool:
    tokens = [t.lower() for t in _TOKEN_RE.findall(line)]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in _SPANISH_STOPWORDS)
    return hits >= stopword_threshold


def _looks_shiwilu(line: str, *, require_apostrophe: bool = True, min_tokens: int = 1) -> bool:
    """Conservative Shiwilu-likely heuristic.

    By default we require at least one apostrophe because Shiwilu uses
    glottal-stop apostrophes pervasively (lau'ker', mu'katapa'su', a'lek'),
    and this is the strongest single signal that separates Shiwilu lines
    from Spanish ones in our raw text. Set require_apostrophe=False to
    relax (will admit more candidates and more Spanish false positives).
    """
    if not line.strip():
        return False
    if _looks_spanish(line):
        return False
    tokens = _TOKEN_RE.findall(line)
    if len(tokens) < min_tokens:
        return False
    if require_apostrophe and "'" not in line:
        return False
    return True


def collect_parallel_shiwilu(parallel_csvs: Iterable[Path]) -> set[str]:
    seen: set[str] = set()
    for csv_path in parallel_csvs:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "source_lang" in df.columns:
            shw_rows = df[df["source_lang"] == "shw"]["source"].astype(str)
            for s in shw_rows:
                seen.add(_normalize_for_compare(s))
        # also consider target column when target_lang == 'shw'
        if "target_lang" in df.columns:
            shw_rows = df[df["target_lang"] == "shw"]["target"].astype(str)
            for s in shw_rows:
                seen.add(_normalize_for_compare(s))
    return seen


def extract_mono_shiwilu(
    parallel_csvs: Sequence[Path],
    candidate_text_paths: Sequence[Path],
    out_path: Path,
    *,
    require_apostrophe: bool = True,
) -> dict:
    parallel_seen = collect_parallel_shiwilu(parallel_csvs)

    mono: set[str] = set()
    skipped_in_parallel = 0
    skipped_looks_spanish = 0
    skipped_no_apostrophe = 0
    candidates_total = 0

    for path in candidate_text_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                candidates_total += 1
                if _normalize_for_compare(line) in parallel_seen:
                    skipped_in_parallel += 1
                    continue
                if _looks_spanish(line):
                    skipped_looks_spanish += 1
                    continue
                if not _looks_shiwilu(line, require_apostrophe=require_apostrophe):
                    if require_apostrophe and "'" not in line:
                        skipped_no_apostrophe += 1
                    continue
                mono.add(line)

    mono_sorted = sorted(mono)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in mono_sorted:
            f.write(s + "\n")

    return {
        "candidates_total": candidates_total,
        "kept": len(mono_sorted),
        "skipped_in_parallel": skipped_in_parallel,
        "skipped_looks_spanish": skipped_looks_spanish,
        "skipped_no_apostrophe": skipped_no_apostrophe,
        "require_apostrophe": require_apostrophe,
        "out_path": str(out_path),
    }


@dataclass(frozen=True)
class BackTranslationConfig:
    bt_cap_x_parallel: float = 2.0


def cap_synthetic(
    synthetic_df: pd.DataFrame,
    parallel_size: int,
    cfg: BackTranslationConfig,
) -> pd.DataFrame:
    cap = int(cfg.bt_cap_x_parallel * parallel_size)
    if len(synthetic_df) <= cap:
        return synthetic_df.reset_index(drop=True)
    # Prefer highest-scored pairs (already filtered above 0.60).
    sorted_df = synthetic_df.sort_values("score", ascending=False).head(cap).reset_index(drop=True)
    return sorted_df


def synthetic_pair_id(idx: int) -> str:
    return f"BT{idx:06d}"


def make_synthetic_dataframe(
    mono_lines: Sequence[str],
    spanish_translations: Sequence[str],
    scores: Sequence[float],
    *,
    accept_threshold: float,
) -> pd.DataFrame:
    if not (len(mono_lines) == len(spanish_translations) == len(scores)):
        raise ValueError("mono / translations / scores must have the same length")

    rows: list[dict] = []
    rejected = 0
    for i, (shw, spa, score) in enumerate(zip(mono_lines, spanish_translations, scores)):
        if score <= accept_threshold:
            rejected += 1
            continue
        pair_id = synthetic_pair_id(i)
        for src_lang, tgt_lang, src, tgt in (
            ("shw", "spa", shw, spa),
            ("spa", "shw", spa, shw),
        ):
            rows.append(
                {
                    "id": f"{pair_id}__{src_lang}2{tgt_lang}",
                    "pair_id": pair_id,
                    "group_id": f"GBT{i:06d}",
                    "source": src,
                    "target": tgt,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "split": "train",
                    "has_audit_flags": False,
                    "origin_source": "backtranslation_v0",
                    "score": float(score),
                    "label": "accepted",
                }
            )
    return pd.DataFrame(rows)
