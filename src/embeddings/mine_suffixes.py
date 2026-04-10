from __future__ import annotations

import json
from collections import Counter

from .config import SPLITS_DIR

TRAIN_PATH = SPLITS_DIR / "train_pairs.jsonl"
OUT_PATH = SPLITS_DIR / "shiwilu_suffixes.json"

MIN_SUFFIX_LEN = 2
MAX_SUFFIX_LEN = 6
MIN_COUNT = 5


def extract_suffixes(word: str) -> list[str]:
    suffixes = []
    for k in range(MIN_SUFFIX_LEN, min(MAX_SUFFIX_LEN, len(word)) + 1):
        suffixes.append(word[-k:])
    return suffixes


def main() -> None:
    counter: Counter[str] = Counter()

    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            shw = row["shiwilu"]
            words = shw.split()
            for word in words:
                if len(word) < MIN_SUFFIX_LEN + 1:
                    continue
                for suffix in extract_suffixes(word):
                    counter[suffix] += 1

    suffixes = [
        {"suffix": s, "count": c}
        for s, c in counter.most_common()
        if c >= MIN_COUNT
    ]

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(suffixes, f, ensure_ascii=False, indent=2)

    print(f"Guardados {len(suffixes)} sufijos candidatos en {OUT_PATH}")


if __name__ == "__main__":
    main()