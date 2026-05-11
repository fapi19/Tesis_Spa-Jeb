"""Phase 7a-bis: round-trip backtranslation with monolingual-Spanish consistency filter.

Pipeline (all on the v0 / v0_xl checkpoint):
    real_spa  --[spa->shw]-->  synth_shw  --[shw->spa]-->  recovered_spa
                                                                |
              SBERT(real_spa, recovered_spa)  ----------------- filter
                                                                |
              accepted pairs  -->  (real_spa, synth_shw)  -->  train_bt_roundtrip.csv

Why this is better than the simple BT in 60_backtranslate.py:
    - Source side is real Spanish (abundant, clean) instead of 76 mono Shiwilu lines.
    - Filter is *monolingual* (real_spa vs recovered_spa) which is more reliable
      than the cross-lingual SBERT score the simple BT uses.

Spanish source: FLORES-200 spa_Latn devtest (1012 sentences). High-quality,
domain-mixed, the standard low-resource MT benchmark. Override with --candidate.

Usage:
    python scripts/nmt/60b_roundtrip_bt.py --variant xl --checkpoint models/nmt/nllb_bidi_lora_v0_xl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.nmt._paths import resolve_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["main", "xl"], default="main")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Override Spanish source. Plain UTF-8 file, one sentence per line.",
    )
    p.add_argument(
        "--source",
        choices=["flores", "wikipedia", "tatoeba", "news_commentary", "opus"],
        default="flores",
        help="Spanish monolingual source to fetch. flores=FLORES-101 (~1k), "
             "wikipedia=es Wikipedia first sentences, tatoeba=Tatoeba es, "
             "news_commentary=News Commentary v18, opus=OPUS-100 es-en sample.",
    )
    p.add_argument(
        "--n-sentences",
        type=int,
        default=3000,
        help="Target number of sentences to fetch from the chosen source (Wikipedia/Tatoeba/news/opus).",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="train_bt_roundtrip",
        help="Output basename inside augmented_dir. Use 'train_bt_roundtrip_iter1' for iter 1, etc.",
    )
    p.add_argument(
        "--accept-threshold",
        type=float,
        default=0.70,
        help="Min SBERT cosine(real_spa, recovered_spa) to accept. "
             "0.70 is stricter than cross-lingual BT default (0.60) since the comparison is monolingual.",
    )
    p.add_argument(
        "--cap-x",
        type=float,
        default=2.0,
        help="Cap accepted synthetic pairs at this multiple of parallel train size.",
    )
    p.add_argument(
        "--max-source-lines",
        type=int,
        default=None,
        help="Truncate Spanish source to this many lines (debug / faster runs).",
    )
    p.add_argument("--report", type=str, default=None)
    return p.parse_args()


def _split_into_sentences_es(text: str) -> list[str]:
    """Naive Spanish sentence splitter (period/?/! followed by space + Capital).
    Good enough for sampling Wikipedia / news intros.
    """
    import re
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", text)
    return [p.strip() for p in parts if p.strip()]


def _fetch_wikipedia_es(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"[phase7a-bis] fetching {n} Spanish sentences from wikimedia/wikipedia (20231101.es) ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    out: list[str] = []
    for article in ds:
        text = article.get("text", "")
        if not text:
            continue
        for sent in _split_into_sentences_es(text)[:3]:
            n_words = len(sent.split())
            if 5 <= n_words <= 30:
                out.append(sent)
                if len(out) >= n:
                    return out
    return out


def _fetch_tatoeba_es(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"[phase7a-bis] fetching {n} Spanish sentences from Tatoeba ...")
    # Tatoeba has spa-eng config among many; we just want the spa side.
    ds = load_dataset("tatoeba", lang1="en", lang2="es", split="train", trust_remote_code=True)
    out: list[str] = []
    for rec in ds:
        t = rec.get("translation") or {}
        v = t.get("es")
        if v and isinstance(v, str) and v.strip():
            out.append(v.strip())
            if len(out) >= n:
                return out
    return out


def _fetch_news_commentary_es(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"[phase7a-bis] fetching {n} Spanish sentences from News Commentary v18 ...")
    for spec in (("Helsinki-NLP/news_commentary", "en-es"), ("news_commentary", "en-es")):
        try:
            ds = load_dataset(spec[0], spec[1], split="train", trust_remote_code=True)
            out: list[str] = []
            for rec in ds:
                t = rec.get("translation") or {}
                v = t.get("es")
                if v and isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    if len(out) >= n:
                        return out
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[phase7a-bis] news_commentary via {spec[0]} failed: {exc}")
    return []


def _fetch_opus100_es(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"[phase7a-bis] fetching {n} Spanish sentences from OPUS-100 (en-es) ...")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split="train", trust_remote_code=True)
    out: list[str] = []
    for rec in ds:
        t = rec.get("translation") or {}
        v = t.get("es")
        if v and isinstance(v, str) and v.strip():
            out.append(v.strip())
            if len(out) >= n * 2:  # over-collect to allow shuffle
                break
    import random as _rnd
    rng = _rnd.Random(2026)
    rng.shuffle(out)
    return out[:n]


def _load_flores_spa_devtest() -> list[str]:
    """FLORES-101 spa devtest. ~1012 sentences, high quality, mixed domain."""
    from datasets import load_dataset

    flores_attempts = (
        ("gsarti/flores_101", "spanish", "devtest", "sentence"),
        ("Muennighoff/flores200", "spa_Latn", "devtest", None),
        ("facebook/flores", "spa_Latn", "devtest", None),
    )
    for spec in flores_attempts:
        repo, cfg, split, field_hint = spec
        try:
            ds = load_dataset(repo, cfg, split=split, trust_remote_code=True)
            field = field_hint or ("text" if "text" in ds.column_names else "sentence")
            lines = [str(x).strip() for x in ds[field] if str(x).strip()]
            if not lines:
                raise RuntimeError(f"loaded but empty: {repo}/{cfg}/{split}")
            print(f"[phase7a-bis] FLORES source: {repo} ({cfg}, {split}) -> {len(lines)} lines")
            return lines
        except Exception as exc:  # noqa: BLE001
            print(f"[phase7a-bis] could not load {repo}: {exc}")

    # Fallback 1: OPUS-100 es-en, take Spanish side, sample 1012.
    try:
        print("[phase7a-bis] FLORES unavailable, falling back to Helsinki-NLP/opus-100 es-en (Spanish side, 1012 sample)")
        ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split="train", trust_remote_code=True)
        # opus-100 has 'translation' dict field
        spa = []
        for rec in ds:
            t = rec.get("translation") or {}
            v = t.get("es")
            if v and isinstance(v, str) and v.strip():
                spa.append(v.strip())
                if len(spa) >= 5000:
                    break
        # Deterministic sample of 1012
        import random as _rnd
        rng = _rnd.Random(2026)
        rng.shuffle(spa)
        lines = spa[:1012]
        print(f"[phase7a-bis] OPUS-100 fallback -> {len(lines)} lines")
        return lines
    except Exception as exc:  # noqa: BLE001
        print(f"[phase7a-bis] OPUS-100 fallback failed: {exc}")

    raise RuntimeError(
        "Could not load any Spanish monolingual source from huggingface. "
        "Pass --candidate <file> with a UTF-8 plaintext Spanish corpus instead."
    )


def _load_source_lines(args: argparse.Namespace) -> list[str]:
    if args.candidate:
        path = Path(args.candidate).resolve()
        with path.open(encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        print(f"[phase7a-bis] custom Spanish source: {path} -> {len(lines)} lines")
    elif args.source == "flores":
        lines = _load_flores_spa_devtest()
    elif args.source == "wikipedia":
        lines = _fetch_wikipedia_es(args.n_sentences)
    elif args.source == "tatoeba":
        lines = _fetch_tatoeba_es(args.n_sentences)
    elif args.source == "news_commentary":
        lines = _fetch_news_commentary_es(args.n_sentences)
    elif args.source == "opus":
        lines = _fetch_opus100_es(args.n_sentences)
    else:
        raise ValueError(f"unknown --source {args.source!r}")
    if args.max_source_lines:
        lines = lines[: args.max_source_lines]
        print(f"[phase7a-bis] truncated to {len(lines)} lines (--max-source-lines)")
    return lines


def main() -> int:
    args = parse_args()
    nmt = resolve_paths(PROJECT_ROOT, args.variant)
    parallel_dir = nmt.filtered_dir
    augmented_dir = nmt.augmented_dir
    suffix = "_xl" if args.variant == "xl" else ""
    reports_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"augmentation{suffix}"

    augmented_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    spa_lines = _load_source_lines(args)
    if not spa_lines:
        print("[phase7a-bis] no Spanish source lines available", file=sys.stderr)
        return 2

    import torch  # noqa: F401
    import yaml
    from src.nmt.inference.generate import GenerationConfig, generate_for_direction, load_checkpoint
    from src.nmt.preprocessing.semantic_filter import (
        SemanticFilterConfig,
        load_embedding_model,
    )

    with (PROJECT_ROOT / "config" / "nmt" / "training.yaml").open(encoding="utf-8") as f:
        training_yaml = yaml.safe_load(f)
    base_model = training_yaml["base_model"]
    lang_code_map = {str(k): str(v) for k, v in training_yaml["data"]["lang_code_map"].items()}

    gen_cfg = GenerationConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "inference.yaml")
    print(f"[phase7a-bis] loading checkpoint {args.checkpoint}")
    model, tokenizer, device = load_checkpoint(Path(args.checkpoint), base_model=base_model, device="auto")

    df_fwd = pd.DataFrame(
        {
            "id": [f"RTBT{i:06d}__spa2shw" for i in range(len(spa_lines))],
            "pair_id": [f"RTBT{i:06d}" for i in range(len(spa_lines))],
            "source": spa_lines,
            "target": [""] * len(spa_lines),
        }
    )
    print(f"[phase7a-bis] FORWARD spa->shw on {len(df_fwd)} lines (beam={gen_cfg.num_beams}) ...")
    fwd_preds = generate_for_direction(
        model, tokenizer, df_fwd,
        src_plan="spa", tgt_plan="shw",
        lang_code_map=lang_code_map, cfg=gen_cfg, device=device, return_topk=False,
    )
    synth_shw = [p["hypothesis"] for p in fwd_preds]

    df_back = pd.DataFrame(
        {
            "id": [f"RTBT{i:06d}__shw2spa" for i in range(len(synth_shw))],
            "pair_id": [f"RTBT{i:06d}" for i in range(len(synth_shw))],
            "source": synth_shw,
            "target": [""] * len(synth_shw),
        }
    )
    print(f"[phase7a-bis] BACK shw->spa on {len(df_back)} lines ...")
    back_preds = generate_for_direction(
        model, tokenizer, df_back,
        src_plan="shw", tgt_plan="spa",
        lang_code_map=lang_code_map, cfg=gen_cfg, device=device, return_topk=False,
    )
    recovered_spa = [p["hypothesis"] for p in back_preds]

    del model
    import torch as _torch
    _torch.cuda.empty_cache()

    print("[phase7a-bis] SBERT scoring real_spa vs recovered_spa (monolingual) ...")
    filter_cfg = SemanticFilterConfig.from_yaml(PROJECT_ROOT / "config" / "nmt" / "filter.yaml", PROJECT_ROOT)
    sbert = load_embedding_model(filter_cfg)
    emb_real = sbert.encode(spa_lines, batch_size=filter_cfg.batch_size, normalize_embeddings=True, show_progress_bar=False)
    emb_rec = sbert.encode(recovered_spa, batch_size=filter_cfg.batch_size, normalize_embeddings=True, show_progress_bar=False)
    scores = (np.asarray(emb_real) * np.asarray(emb_rec)).sum(axis=1).tolist()

    rows: list[dict] = []
    rejected = 0
    for i, (real_spa, shw, rec_spa, score) in enumerate(zip(spa_lines, synth_shw, recovered_spa, scores)):
        if score <= args.accept_threshold:
            rejected += 1
            continue
        if not shw.strip() or not real_spa.strip():
            rejected += 1
            continue
        pair_id = f"RTBT{i:06d}"
        for src_lang, tgt_lang, src, tgt in (
            ("spa", "shw", real_spa, shw),
            ("shw", "spa", shw, real_spa),
        ):
            rows.append(
                {
                    "id": f"{pair_id}__{src_lang}2{tgt_lang}",
                    "pair_id": pair_id,
                    "group_id": f"GRTBT{i:06d}",
                    "source": src,
                    "target": tgt,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "split": "train",
                    "has_audit_flags": False,
                    "origin_source": "backtranslation_roundtrip_v0",
                    "score": float(score),
                    "label": "accepted",
                }
            )

    syn_df = pd.DataFrame(rows)
    n_pairs_kept = len(syn_df) // 2
    print(f"[phase7a-bis] accepted {n_pairs_kept} / {len(spa_lines)} pairs at threshold {args.accept_threshold} (rejected {rejected})")

    parallel_train = pd.read_csv(parallel_dir / "train.csv", encoding="utf-8-sig")
    cap = int(args.cap_x * len(parallel_train))
    cap_rows = cap * 2  # each pair -> 2 directional rows
    if len(syn_df) > cap_rows:
        syn_df = syn_df.sort_values("score", ascending=False).head(cap_rows).reset_index(drop=True)
        print(f"[phase7a-bis] capped to {cap} pairs ({len(syn_df)} rows)")

    out_csv = augmented_dir / f"{args.output_name}.csv"
    syn_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[phase7a-bis] wrote {out_csv.relative_to(PROJECT_ROOT)}")

    score_arr = np.asarray(scores, dtype=float)
    report = {
        "phase": "7a-bis",
        "step": "roundtrip_backtranslation",
        "variant": args.variant,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "spanish_source_lines": len(spa_lines),
        "accept_threshold": args.accept_threshold,
        "cap_x_parallel": args.cap_x,
        "parallel_train_size": int(len(parallel_train)),
        "pairs_accepted": n_pairs_kept,
        "directional_rows_written": int(len(syn_df)),
        "score_stats": {
            "mean": float(score_arr.mean()) if score_arr.size else None,
            "std": float(score_arr.std()) if score_arr.size else None,
            "min": float(score_arr.min()) if score_arr.size else None,
            "max": float(score_arr.max()) if score_arr.size else None,
            "p10": float(np.percentile(score_arr, 10)) if score_arr.size else None,
            "p50": float(np.percentile(score_arr, 50)) if score_arr.size else None,
            "p90": float(np.percentile(score_arr, 90)) if score_arr.size else None,
        },
    }
    out_dir = Path(args.report) if args.report else reports_dir
    (out_dir / "backtranslation_roundtrip.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[phase7a-bis] report -> {(out_dir / 'backtranslation_roundtrip.json').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
