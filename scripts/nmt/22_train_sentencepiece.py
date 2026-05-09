"""Phase 3 runner: train SentencePiece Unigram (vocab=8000, cc=1.0) and emit
a side-by-side tokenization comparison against the NLLB tokenizer.

Outputs:
    data/processed/05_nmt_canonical/all_text_for_sp_nmt.txt
    models/nmt/sentencepiece/sp_unigram_8k.{model,vocab}
    reports/05_nmt/preprocessing/sentencepiece_stats.json
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import sentencepiece as spm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.nmt.preprocessing.train_sentencepiece import (  # noqa: E402
    SentencePieceConfig,
    collect_corpus,
    train_sentencepiece,
    write_corpus,
)

CANON_DIR = PROJECT_ROOT / "data" / "processed" / "05_nmt_canonical"
FILTERED_DIR = PROJECT_ROOT / "data" / "processed" / "06_nmt_filtered"
SP_OUT_DIR = PROJECT_ROOT / "models" / "nmt" / "sentencepiece"
REPORTS_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "preprocessing"

NLLB_BASE = "facebook/nllb-200-distilled-600M"
SAMPLE_SIZE = 50
SAMPLE_SEED = 42


def shiwilu_sample(filtered_train_csv: Path, n: int, seed: int) -> list[str]:
    df = pd.read_csv(filtered_train_csv, encoding="utf-8-sig")
    shw = df[df["source_lang"] == "shw"][["pair_id", "source"]].drop_duplicates(subset=["pair_id"])
    shw = shw[shw["source"].notna()]
    shw = shw[shw["source"].str.strip().astype(bool)]
    rng = random.Random(seed)
    pool = shw["source"].tolist()
    rng.shuffle(pool)
    return pool[:n]


def main() -> int:
    # 1) Build the SP corpus from accepted train.
    sentences = collect_corpus(FILTERED_DIR / "train.csv")
    corpus_path = CANON_DIR / "all_text_for_sp_nmt.txt"
    n_lines = write_corpus(sentences, corpus_path)
    print(f"[phase3] corpus -> {corpus_path.relative_to(PROJECT_ROOT)} ({n_lines} lines)")

    # 2) Train SentencePiece.
    cfg = SentencePieceConfig(vocab_size=8000, char_coverage=1.0, model_type="unigram")
    sp_model_path, sp_vocab_path = train_sentencepiece(corpus_path, SP_OUT_DIR, cfg)
    print(f"[phase3] sp model -> {sp_model_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase3] sp vocab -> {sp_vocab_path.relative_to(PROJECT_ROOT)}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model_path))
    sp_vocab_size = sp.get_piece_size()

    # 3) Side-by-side tokenization on Shiwilu samples vs NLLB tokenizer.
    print(f"[phase3] loading NLLB tokenizer ({NLLB_BASE}) for comparison ...")
    nllb_tok = AutoTokenizer.from_pretrained(NLLB_BASE)

    samples = shiwilu_sample(FILTERED_DIR / "train.csv", SAMPLE_SIZE, SAMPLE_SEED)

    comparisons: list[dict] = []
    sp_token_counts: list[int] = []
    nllb_token_counts: list[int] = []
    for sentence in samples:
        sp_tokens = sp.encode(sentence, out_type=str)
        nllb_tokens = nllb_tok.tokenize(sentence)
        sp_token_counts.append(len(sp_tokens))
        nllb_token_counts.append(len(nllb_tokens))
        comparisons.append(
            {
                "sentence": sentence,
                "sp_unigram_8k": {
                    "tokens": sp_tokens,
                    "n": len(sp_tokens),
                },
                "nllb": {
                    "tokens": nllb_tokens,
                    "n": len(nllb_tokens),
                },
            }
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 3,
        "step": "train_sentencepiece",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {
            "vocab_size": cfg.vocab_size,
            "char_coverage": cfg.char_coverage,
            "model_type": cfg.model_type,
        },
        "corpus": {
            "path": str(corpus_path.relative_to(PROJECT_ROOT)),
            "lines": n_lines,
        },
        "sentencepiece": {
            "model_path": str(sp_model_path.relative_to(PROJECT_ROOT)),
            "vocab_path": str(sp_vocab_path.relative_to(PROJECT_ROOT)),
            "actual_vocab_size": sp_vocab_size,
        },
        "comparison": {
            "nllb_base": NLLB_BASE,
            "sample_size": len(samples),
            "sample_seed": SAMPLE_SEED,
            "sp_unigram_avg_tokens": (
                sum(sp_token_counts) / len(sp_token_counts) if sp_token_counts else 0.0
            ),
            "nllb_avg_tokens": (
                sum(nllb_token_counts) / len(nllb_token_counts) if nllb_token_counts else 0.0
            ),
            "examples": comparisons,
        },
        "rationale": (
            "Plan.md sections 15-17 mandate SP Unigram for vocabulary / morphology "
            "analysis. NLLB ships its own tokenizer used in Phase 4; this artifact "
            "supports the agglutinative-friendly tokenization argument and serves "
            "as a comparison fixture, not the runtime tokenizer."
        ),
    }
    out_path = REPORTS_DIR / "sentencepiece_stats.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase3] report -> {out_path.relative_to(PROJECT_ROOT)}")
    print(
        f"[phase3] avg tokens on Shiwilu: SP unigram = "
        f"{report['comparison']['sp_unigram_avg_tokens']:.2f}, "
        f"NLLB = {report['comparison']['nllb_avg_tokens']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
