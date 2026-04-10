from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

from .config import RAW_CSV, SPLITS_DIR

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("’", "'").replace("`", "'")
    text = " ".join(text.split())
    return text


def load_parallel_csv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shw = normalize_text(row["SHIWILU_normalizado"])
            es = normalize_text(row["ESP_normalizado"])
            if shw and es:
                rows.append((shw, es))
    return rows


def save_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    pairs = load_parallel_csv(RAW_CSV)
    if not pairs:
        raise ValueError("No se encontraron pares válidos.")

    split = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    with (SPLITS_DIR / "all_text.txt").open("w", encoding="utf-8") as f:
        for shw, es in pairs:
            f.write(shw + "\n")
            f.write(es + "\n")

    save_jsonl(
        SPLITS_DIR / "train_pairs.jsonl",
        [{"shiwilu": shw, "spanish": es} for shw, es in train_pairs],
    )
    save_jsonl(
        SPLITS_DIR / "val_pairs.jsonl",
        [{"shiwilu": shw, "spanish": es} for shw, es in val_pairs],
    )

    print(f"Total pares: {len(pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")


if __name__ == "__main__":
    main()