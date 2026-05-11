"""Phase 8b runner: build the anonymized human-evaluation template.

Stratified random sample of N items per direction, balancing source
distribution (flashcards vs pdf_textos) and length buckets. The hypothesis
columns are anonymized via a random model-letter mapping so reviewers
cannot bias scores by knowing which system produced which text.

Outputs:
    reports/05_nmt/evaluation/human_eval_template.csv
    reports/05_nmt/evaluation/human_eval_anon_key.json   (the secret mapping)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import string
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from scripts.nmt._paths import resolve_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["main", "xl"], default="main")
    p.add_argument("--per-direction", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--v0", default=None, help="Default: nllb_bidi_lora_v0[_xl]")
    p.add_argument("--v1", default=None, help="Default: nllb_bidi_lora_v1_bt[_xl]")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    return p.parse_args()


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _length_bucket(text: str) -> str:
    n = len(str(text).split())
    if n <= 5:
        return "short"
    if n <= 12:
        return "medium"
    return "long"


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    df["len_bucket"] = df["source"].astype(str).apply(_length_bucket)
    if "origin_source" not in df.columns:
        df["origin_source"] = "unknown"

    strata = list(df.groupby(["origin_source", "len_bucket"]))
    total = sum(len(g) for _, g in strata)
    if total == 0:
        return df.head(0)

    quotas: list[tuple[tuple[str, str], int]] = []
    remainder = 0
    for key, group in strata:
        quota = (n * len(group)) / total
        floor = int(quota)
        remainder += quota - floor
        quotas.append((key, floor))

    # Distribute fractional remainders deterministically.
    extras_to_assign = round(remainder)
    quotas.sort(key=lambda kv: -((n * len(dict(strata)[kv[0]])) / total - kv[1]))
    for i in range(extras_to_assign):
        if i >= len(quotas):
            break
        quotas[i] = (quotas[i][0], quotas[i][1] + 1)

    rows: list[pd.Series] = []
    for key, k in quotas:
        if k <= 0:
            continue
        group = dict(strata)[key]
        chosen = group.sample(n=min(k, len(group)), random_state=seed)
        rows.append(chosen)
    sampled = pd.concat(rows, ignore_index=True) if rows else df.head(0)

    # If under-sampled (e.g., uneven strata), top up randomly without replacement.
    if len(sampled) < n:
        extra_pool = df[~df["id"].isin(set(sampled["id"]))]
        topup = extra_pool.sample(n=min(n - len(sampled), len(extra_pool)), random_state=seed + 1)
        sampled = pd.concat([sampled, topup], ignore_index=True)

    return sampled.head(n).reset_index(drop=True)


def _hypothesis_for(predictions: dict[str, dict], row_id: str) -> str:
    rec = predictions.get(row_id)
    if rec is None:
        return ""
    return str(rec.get("hypothesis", ""))


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    nmt = resolve_paths(PROJECT_ROOT, args.variant)
    suffix = "_xl" if args.variant == "xl" else ""
    eval_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"evaluation{suffix}"
    rerank_dir = PROJECT_ROOT / "reports" / "05_nmt" / f"reranking{suffix}"
    v0_name = args.v0 or f"nllb_bidi_lora_v0{suffix}"
    v1_name = args.v1 or f"nllb_bidi_lora_v1_bt{suffix}"

    test_csv = nmt.filtered_dir / f"{args.split}.csv"
    test_df = pd.read_csv(test_csv, encoding="utf-8-sig")
    test_df = test_df.dropna(subset=["source", "target"]).reset_index(drop=True)

    v0_eval = _read_jsonl(eval_dir / v0_name / f"{args.split}_predictions.jsonl")
    v0_rerank = _read_jsonl(rerank_dir / v0_name / f"{args.split}_predictions_reranked.jsonl")
    v1_eval = _read_jsonl(eval_dir / v1_name / f"{args.split}_predictions.jsonl")
    v1_rerank = _read_jsonl(rerank_dir / v1_name / f"{args.split}_predictions_reranked.jsonl")

    sources = {
        v0_name: v0_eval,
        f"{v0_name}_reranked": v0_rerank,
        v1_name: v1_eval,
        f"{v1_name}_reranked": v1_rerank,
    }
    available = {k: bool(v) for k, v in sources.items()}
    print(f"[phase8b] available prediction files: {available}")

    # Anonymized model-letter mapping: a random permutation of A..D.
    keys = list(sources.keys())
    rng.shuffle(keys)
    letter_to_source = dict(zip(string.ascii_uppercase[: len(keys)], keys))
    source_to_letter = {v: k for k, v in letter_to_source.items()}

    samples: list[pd.DataFrame] = []
    for direction in ("shw2spa", "spa2shw"):
        sub = test_df[
            (test_df["source_lang"] == direction.split("2")[0])
            & (test_df["target_lang"] == direction.split("2")[1])
        ].copy()
        sample = _stratified_sample(sub, args.per_direction, args.seed)
        sample["direction"] = direction
        samples.append(sample)
    sampled_df = pd.concat(samples, ignore_index=True)

    # Build the template rows.
    template_rows: list[dict] = []
    for _, row in sampled_df.iterrows():
        rid = row["id"]
        item = {
            "id": rid,
            "pair_id": row["pair_id"],
            "direction": row["direction"],
            "origin_source": row.get("origin_source", "unknown"),
            "source": row["source"],
            "reference": row["target"],
        }
        for letter, src_key in letter_to_source.items():
            item[f"hypothesis_{letter}"] = _hypothesis_for(sources[src_key], rid)
        item.update(
            {
                "adequacy_1_5": "",
                "fluency_1_5": "",
                "cultural_relevance_1_5": "",
                "notes": "",
            }
        )
        template_rows.append(item)

    template_df = pd.DataFrame(template_rows)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / "human_eval_template.csv"
    template_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[phase8b] wrote {out_csv.relative_to(PROJECT_ROOT)} ({len(template_df)} rows)")

    key = {
        "phase": "8b",
        "step": "human_eval_template",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "per_direction": args.per_direction,
        "anonymization": {
            "letter_to_source": letter_to_source,
            "source_to_letter": source_to_letter,
        },
        "rubric": {
            "adequacy_1_5": "5 = preserves all meaning, 1 = loses all meaning",
            "fluency_1_5": "5 = native-speaker fluent, 1 = ungrammatical",
            "cultural_relevance_1_5": "5 = idiomatically appropriate, 1 = jarring or wrong register",
        },
        "sources_available": available,
        "stratification": "balanced by origin_source (flashcards vs pdf_textos) and length bucket (short<=5, medium<=12, long)",
    }
    key_path = eval_dir / "human_eval_anon_key.json"
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[phase8b] wrote {key_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
