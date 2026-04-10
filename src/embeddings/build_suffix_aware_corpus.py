from __future__ import annotations

import json
from pathlib import Path

from .config import SPLITS_DIR

INPUT_JSONL = SPLITS_DIR / "train_pairs.jsonl"
VAL_JSONL = SPLITS_DIR / "val_pairs.jsonl"
SUFFIXES_JSON = SPLITS_DIR / "shiwilu_suffixes.json"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUFFIX_COUNT = 10


def load_suffixes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    suffixes = [x["suffix"] for x in data if x["count"] >= MIN_SUFFIX_COUNT]
    suffixes = sorted(set(suffixes), key=len, reverse=True)
    return suffixes


def split_suffixes(word: str, suffixes: list[str]) -> str:
    for suffix in suffixes:
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            stem = word[: -len(suffix)]
            return f"{stem} @@{suffix}"
    return word


def transform_shiwilu(text: str, suffixes: list[str]) -> str:
    return " ".join(split_suffixes(w, suffixes) for w in text.split())


def transform_jsonl(input_path: Path, output_path: Path, suffixes: list[str]) -> None:
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            row["shiwilu"] = transform_shiwilu(row["shiwilu"], suffixes)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_all_text(train_path: Path, val_path: Path, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f_out:
        for p in [train_path, val_path]:
            with p.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    row = json.loads(line)
                    f_out.write(row["shiwilu"] + "\n")
                    f_out.write(row["spanish"] + "\n")


def main() -> None:
    suffixes = load_suffixes(SUFFIXES_JSON)

    train_out = SPLITS_DIR / "train_pairs_suffix_aware.jsonl"
    val_out = SPLITS_DIR / "val_pairs_suffix_aware.jsonl"
    all_text_out = SPLITS_DIR / "all_text_suffix_aware.txt"

    transform_jsonl(INPUT_JSONL, train_out, suffixes)
    transform_jsonl(VAL_JSONL, val_out, suffixes)
    build_all_text(train_out, val_out, all_text_out)

    print("Corpus suffix-aware generado.")


if __name__ == "__main__":
    main()