from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_CSV = PROJECT_ROOT / "data/processed/03_pre_embeddings/dataset_pre_embeddings.csv"
SPLITS_DIR = PROJECT_ROOT / "data/processed/04_splits"
REPORT_DIR = PROJECT_ROOT / "reports/nmt"

SEED = 42
NARRATIVE_THRESHOLD = 25
RATIO_THRESHOLD = 4.0

FANCY_DOUBLE = re.compile(r"[\u201c\u201d\u201e\u201f\u00ab\u00bb]")
FANCY_SINGLE = re.compile(r"[\u2018\u2019\u201a\u201b]")
MULTI_SPACE = re.compile(r" {2,}")
REPEATED_PUNCT = re.compile(r"([.!?,:;])\1+")
PARENS = re.compile(r"\(.*?\)")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = FANCY_DOUBLE.sub('"', text)
    text = FANCY_SINGLE.sub("'", text)
    text = MULTI_SPACE.sub(" ", text)
    text = REPEATED_PUNCT.sub(r"\1", text)
    if not text[:1] in ("¡", "¿"):
        text = text[0].lower() + text[1:] if text else text
    return text


def has_editorial_markers(shw: str, spa: str) -> bool:
    return bool(PARENS.search(spa) or PARENS.search(shw))


def word_count(text: str) -> int:
    return len(text.split())


def length_ratio(src: str, tgt: str) -> float:
    a, b = word_count(src), word_count(tgt)
    return max(a, b) / max(1, min(a, b))


def load_csv(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shw = normalize_text(row["SHIWILU_normalizado"])
            spa = normalize_text(row["ESP_normalizado"])
            if shw and spa:
                pairs.append({"shiwilu": shw, "spanish": spa})
    return pairs


def write_jsonl(pairs: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def clean_and_split() -> None:
    raw = load_csv(RAW_CSV)
    total_original = len(raw)
    print(f"Pares cargados del CSV: {total_original}")

    # --- Editorial / glosas --------------------------------------------------
    editorial: list[dict[str, str]] = []
    non_editorial: list[dict[str, str]] = []
    for p in raw:
        if has_editorial_markers(p["shiwilu"], p["spanish"]):
            editorial.append(p)
        else:
            non_editorial.append(p)
    print(f"Editorial/glosas: {len(editorial)}")

    # --- Dedup exact ---------------------------------------------------------
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    n_dupes = 0
    for p in non_editorial:
        key = (p["shiwilu"], p["spanish"])
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        deduped.append(p)
    print(f"Duplicados exactos eliminados: {n_dupes}")

    # --- Conflicts -----------------------------------------------------------
    by_shw: dict[str, list[dict[str, str]]] = defaultdict(list)
    for p in deduped:
        by_shw[p["shiwilu"]].append(p)

    conflicts: list[dict[str, str]] = []
    no_conflict: list[dict[str, str]] = []
    n_conflict_sources = 0
    for shw, variants in by_shw.items():
        unique_spa = {v["spanish"] for v in variants}
        if len(unique_spa) >= 2:
            n_conflict_sources += 1
            conflicts.extend(variants)
            canonical = min(unique_spa, key=len)
            no_conflict.append({"shiwilu": shw, "spanish": canonical})
        else:
            no_conflict.append(variants[0])
    print(f"Fuentes con conflicto: {n_conflict_sources} ({len(conflicts)} pares)")

    # --- Length filter --------------------------------------------------------
    imbalanced: list[dict[str, str]] = []
    clean_pairs: list[dict[str, str]] = []

    for p in no_conflict:
        wc_shw = word_count(p["shiwilu"])
        wc_spa = word_count(p["spanish"])

        if wc_shw == 0 or wc_spa == 0:
            continue

        if length_ratio(p["shiwilu"], p["spanish"]) > RATIO_THRESHOLD:
            editorial.append(p)
            imbalanced.append(p)
            continue

        clean_pairs.append(p)

    print(f"Desbalance extremo -> editorial: {len(imbalanced)}")
    print(f"Pares limpios totales: {len(clean_pairs)}")

    # --- Shuffle + split 80/10/10 --------------------------------------------
    random.seed(SEED)
    random.shuffle(clean_pairs)

    n = len(clean_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = clean_pairs[:n_train]
    val = clean_pairs[n_train : n_train + n_val]
    test = clean_pairs[n_train + n_val :]

    print(f"\nSplits (seed={SEED}):")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")

    # --- Write splits --------------------------------------------------------
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(train, SPLITS_DIR / "train_clean.jsonl")
    write_jsonl(val, SPLITS_DIR / "val_clean.jsonl")
    write_jsonl(test, SPLITS_DIR / "test_clean.jsonl")
    write_jsonl(conflicts, SPLITS_DIR / "train_conflicts.jsonl")
    write_jsonl(editorial, SPLITS_DIR / "train_editorial.jsonl")

    # --- all_text_clean.txt for tokenizer ------------------------------------
    with (SPLITS_DIR / "all_text_clean.txt").open("w", encoding="utf-8") as f:
        for p in clean_pairs:
            f.write(p["shiwilu"] + "\n")
            f.write(p["spanish"] + "\n")
    print(f"\nall_text_clean.txt: {len(clean_pairs) * 2} lineas")

    # --- Report --------------------------------------------------------------
    report = {
        "total_original": total_original,
        "duplicates_removed": n_dupes,
        "editorial_glosas": len(editorial),
        "conflicts_sources": n_conflict_sources,
        "conflicts_pairs": len(conflicts),
        "imbalanced_to_editorial": len(imbalanced),
        "clean_total": len(clean_pairs),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "seed": SEED,
        "thresholds": {
            "ratio": RATIO_THRESHOLD,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "clean_splits_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReporte: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    clean_and_split()


if __name__ == "__main__":
    main()
