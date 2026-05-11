from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import (
    RAW_CSV,
    resolve_preprocessing_report_dir,
    resolve_splits_dir,
)


TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
DEFAULT_SEED = 42

APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "`": "'",
        "´": "'",
        "ʼ": "'",
        "ʹ": "'",
    }
)
DOUBLE_QUOTE_RE = re.compile(r'["“”„‟]')
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    source_pair_id: str
    source: str
    raw_shiwilu: str
    raw_spanish: str
    normalized_shiwilu: str
    normalized_spanish: str
    has_audit_flags: bool
    group_id: str = ""

    @property
    def shiwilu(self) -> str:
        return self.normalized_shiwilu

    @property
    def spanish(self) -> str:
        return self.normalized_spanish


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def normalize_text(text: str, *, language: str) -> str:
    """Normalize conservatively without breaking Shiwlu-internal morphology."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.translate(APOSTROPHE_TRANSLATION)
    normalized = DOUBLE_QUOTE_RE.sub("", normalized)
    normalized = normalized.strip().lower()
    normalized = normalized.replace(" ,", ",")
    normalized = SPACE_RE.sub(" ", normalized)
    return normalized


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "sí", "si"}


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[PairRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for idx, row in enumerate(reader):
            raw_shiwilu = row.get("SHIWILU_original", "")
            raw_spanish = row.get("ESP_original", "")
            shiwilu_source = row.get("SHIWILU_normalizado", raw_shiwilu)
            spanish_source = row.get("ESP_normalizado", raw_spanish)
            records.append(
                PairRecord(
                    pair_id=row.get("pair_id") or f"P{idx:05d}",
                    source_pair_id=row.get("source_pair_id", ""),
                    source=row.get("source", "unknown"),
                    raw_shiwilu=raw_shiwilu,
                    raw_spanish=raw_spanish,
                    normalized_shiwilu=normalize_text(shiwilu_source, language="shiwilu"),
                    normalized_spanish=normalize_text(spanish_source, language="spanish"),
                    has_audit_flags=parse_bool(row.get("has_audit_flags", "false")),
                )
            )
    return records


def deduplicate_exact(records: Iterable[PairRecord]) -> tuple[list[PairRecord], list[PairRecord]]:
    seen: set[tuple[str, str]] = set()
    kept = []
    duplicates = []
    for record in records:
        key = (record.normalized_shiwilu, record.normalized_spanish)
        if key in seen:
            duplicates.append(record)
            continue
        seen.add(key)
        kept.append(record)
    return kept, duplicates


def split_empty_records(records: Iterable[PairRecord]) -> tuple[list[PairRecord], list[PairRecord]]:
    kept = []
    empty = []
    for record in records:
        if record.normalized_shiwilu and record.normalized_spanish:
            kept.append(record)
        else:
            empty.append(record)
    return kept, empty


def assign_group_ids(records: list[PairRecord]) -> list[PairRecord]:
    union_find = UnionFind(len(records))
    shiwilu_seen: dict[str, int] = {}
    spanish_seen: dict[str, int] = {}

    for idx, record in enumerate(records):
        if record.normalized_shiwilu in shiwilu_seen:
            union_find.union(idx, shiwilu_seen[record.normalized_shiwilu])
        else:
            shiwilu_seen[record.normalized_shiwilu] = idx

        if record.normalized_spanish in spanish_seen:
            union_find.union(idx, spanish_seen[record.normalized_spanish])
        else:
            spanish_seen[record.normalized_spanish] = idx

    roots = sorted({union_find.find(idx) for idx in range(len(records))})
    group_ids = {root: f"G{group_idx:05d}" for group_idx, root in enumerate(roots)}
    return [
        PairRecord(
            pair_id=record.pair_id,
            source_pair_id=record.source_pair_id,
            source=record.source,
            raw_shiwilu=record.raw_shiwilu,
            raw_spanish=record.raw_spanish,
            normalized_shiwilu=record.normalized_shiwilu,
            normalized_spanish=record.normalized_spanish,
            has_audit_flags=record.has_audit_flags,
            group_id=group_ids[union_find.find(idx)],
        )
        for idx, record in enumerate(records)
    ]


# Sources whose nature is vocabulary-level (word/short-phrase lookups, not
# grammatical exercises). All their entries are forced to the train split with
# downstream loss weighting in the NMT trainer; they never appear in valid/test.
LEXICAL_SOURCES: frozenset[str] = frozenset({"extra", "cotidianas"})


def split_by_group(
    records: list[PairRecord],
    *,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, list[PairRecord]]:
    # Group records by group_id first. If ANY record in a group comes from a
    # lexical source, the whole group is forced to train (preserving group
    # integrity while keeping lexical entries out of valid/test). Pure
    # non-lexical groups participate in the standard random 80/10/10 split.
    groups: dict[str, list[PairRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)

    lexical_groups: dict[str, list[PairRecord]] = {}
    main_groups: dict[str, list[PairRecord]] = {}
    for gid, group_records in groups.items():
        if any(r.source in LEXICAL_SOURCES for r in group_records):
            lexical_groups[gid] = group_records
        else:
            main_groups[gid] = group_records

    main_group_items = list(main_groups.items())
    rng = random.Random(seed)
    rng.shuffle(main_group_items)

    # Targets calculated over the main (non-lexical) records so the
    # split ratios reflect the evaluable corpus.
    total_main = sum(len(g) for g in main_groups.values())
    train_target = int(total_main * train_ratio)
    valid_target = int(total_main * valid_ratio)

    splits: dict[str, list[PairRecord]] = {"train": [], "valid": [], "test": []}
    for _, group_records in main_group_items:
        if len(splits["train"]) < train_target:
            splits["train"].extend(group_records)
        elif len(splits["valid"]) < valid_target:
            splits["valid"].extend(group_records)
        else:
            splits["test"].extend(group_records)

    # Append all lexical groups (and any non-lexical pairs sharing their
    # group_id) to train. This preserves group integrity for the canonical
    # NMT pipeline while keeping the test set sentence-level only.
    for _, group_records in lexical_groups.items():
        splits["train"].extend(group_records)

    return splits


def record_to_json(record: PairRecord) -> dict[str, object]:
    return {
        "pair_id": record.pair_id,
        "group_id": record.group_id,
        "source": record.source,
        "source_pair_id": record.source_pair_id,
        "shiwilu": record.shiwilu,
        "spanish": record.spanish,
        "raw_shiwilu": record.raw_shiwilu,
        "raw_spanish": record.raw_spanish,
        "normalized_shiwilu": record.normalized_shiwilu,
        "normalized_spanish": record.normalized_spanish,
        "has_audit_flags": record.has_audit_flags,
    }


def record_to_csv(record: PairRecord) -> dict[str, object]:
    return {
        "pair_id": record.pair_id,
        "group_id": record.group_id,
        "ESP_original": record.raw_spanish,
        "SHIWILU_original": record.raw_shiwilu,
        "ESP_normalizado": record.normalized_spanish,
        "SHIWILU_normalizado": record.normalized_shiwilu,
        "source": record.source,
        "source_pair_id": record.source_pair_id,
        "has_audit_flags": record.has_audit_flags,
    }


def write_jsonl(path: Path, records: Iterable[PairRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record_to_json(record), ensure_ascii=False) + "\n")


def write_csv(path: Path, records: Iterable[PairRecord]) -> None:
    rows = [record_to_csv(record) for record in records]
    fieldnames = [
        "pair_id",
        "group_id",
        "ESP_original",
        "SHIWILU_original",
        "ESP_normalizado",
        "SHIWILU_normalizado",
        "source",
        "source_pair_id",
        "has_audit_flags",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_all_text(path: Path, records: Iterable[PairRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.shiwilu + "\n")
            f.write(record.spanish + "\n")


def source_distribution(records: Iterable[PairRecord]) -> dict[str, int]:
    return dict(Counter(record.source for record in records))


def group_stats(records: list[PairRecord]) -> dict[str, int]:
    shiwilu_counts = Counter(record.normalized_shiwilu for record in records)
    spanish_counts = Counter(record.normalized_spanish for record in records)
    group_counts = Counter(record.group_id for record in records)
    return {
        "groups": len(group_counts),
        "multi_pair_groups": sum(1 for count in group_counts.values() if count > 1),
        "many_spanish_per_shiwilu": sum(1 for count in shiwilu_counts.values() if count > 1),
        "many_shiwilu_per_spanish": sum(1 for count in spanish_counts.values() if count > 1),
    }


def write_manifest(
    path: Path,
    *,
    variant: str,
    splits_dir: Path,
    input_path: Path,
    seed: int,
    original_count: int,
    included_records: list[PairRecord],
    splits: dict[str, list[PairRecord]],
    excluded: dict[str, list[PairRecord]],
    started_at: datetime,
) -> None:
    elapsed = datetime.now(timezone.utc) - started_at
    manifest = {
        "pipeline": "preprocess_embeddings",
        "variant": variant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": stable_file_hash(input_path),
        "seed": seed,
        "ratios": {
            "train": TRAIN_RATIO,
            "valid": VALID_RATIO,
            "test": 1.0 - TRAIN_RATIO - VALID_RATIO,
        },
        "normalization": {
            "unicode": "NFC",
            "lowercase": True,
            "collapse_spaces": True,
            "normalize_apostrophes_to_ascii": True,
            "preserve_internal_shiwilu_apostrophes": True,
            "remove_double_quotes": True,
            "remove_punctuation": False,
        },
        "counts": {
            "original": original_count,
            "included": len(included_records),
            "excluded_total": sum(len(rows) for rows in excluded.values()),
            "excluded_by_reason": {reason: len(rows) for reason, rows in excluded.items()},
        },
        "groups": group_stats(included_records),
        "splits": {
            name: {
                "rows": len(rows),
                "groups": len({record.group_id for record in rows}),
                "source_distribution": source_distribution(rows),
                "audit_flagged_rows": sum(record.has_audit_flags for record in rows),
            }
            for name, rows in splits.items()
        },
        "artifacts": {
            "canonical_jsonl": {
                "train": str(splits_dir / "train.jsonl"),
                "valid": str(splits_dir / "valid.jsonl"),
                "test": str(splits_dir / "test.jsonl"),
            },
            "sentence_transformers_csv": {
                "train": str(splits_dir / "train.csv"),
                "valid": str(splits_dir / "valid.csv"),
                "test": str(splits_dir / "test.csv"),
            },
            "sentencepiece_text": str(splits_dir / "all_text_for_sp.txt"),
        },
        "elapsed_seconds": elapsed.total_seconds(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def save_outputs(
    splits: dict[str, list[PairRecord]],
    excluded: dict[str, list[PairRecord]],
    *,
    variant: str,
    splits_dir: Path,
    reports_preprocessing_dir: Path,
    input_path: Path,
    seed: int,
    original_count: int,
    started_at: datetime,
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    reports_preprocessing_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(splits_dir / "train.jsonl", splits["train"])
    write_jsonl(splits_dir / "valid.jsonl", splits["valid"])
    write_jsonl(splits_dir / "test.jsonl", splits["test"])

    write_csv(splits_dir / "train.csv", splits["train"])
    write_csv(splits_dir / "valid.csv", splits["valid"])
    write_csv(splits_dir / "test.csv", splits["test"])

    write_all_text(splits_dir / "all_text_for_sp.txt", splits["train"])

    for reason, records in excluded.items():
        write_jsonl(splits_dir / f"excluded_{reason}.jsonl", records)

    included = splits["train"] + splits["valid"] + splits["test"]
    write_manifest(
        reports_preprocessing_dir / "preprocess_manifest.json",
        variant=variant,
        splits_dir=splits_dir,
        input_path=input_path,
        seed=seed,
        original_count=original_count,
        included_records=included,
        splits=splits,
        excluded=excluded,
        started_at=started_at,
    )


def preprocess_embeddings(
    input_path: Path = RAW_CSV,
    *,
    variant: str = "main",
    seed: int = DEFAULT_SEED,
) -> dict[str, list[PairRecord]]:
    splits_dir = resolve_splits_dir(variant)
    reports_preprocessing_dir = resolve_preprocessing_report_dir(variant)
    started_at = datetime.now(timezone.utc)
    records = load_records(input_path)
    non_empty, empty = split_empty_records(records)
    deduped, duplicates = deduplicate_exact(non_empty)
    grouped = assign_group_ids(deduped)
    splits = split_by_group(grouped, seed=seed, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO)
    excluded = {
        "empty_text": empty,
        "exact_duplicate": duplicates,
    }

    save_outputs(
        splits,
        excluded,
        variant=variant,
        splits_dir=splits_dir,
        reports_preprocessing_dir=reports_preprocessing_dir,
        input_path=input_path,
        seed=seed,
        original_count=len(records),
        started_at=started_at,
    )
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocesamiento canónico para embeddings Shiwlu-español.")
    parser.add_argument("--input", type=Path, default=RAW_CSV)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--variant", choices=["main", "xl"], default="main")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = preprocess_embeddings(args.input, variant=args.variant, seed=args.seed)
    total = sum(len(rows) for rows in splits.values())
    print("Preprocesamiento canónico de embeddings completado.")
    print(f"Variante: {args.variant}")
    print(f"Total incluido: {total}")
    print(f"Train: {len(splits['train'])}")
    print(f"Valid: {len(splits['valid'])}")
    print(f"Test: {len(splits['test'])}")
    print(
        "Manifiesto: "
        f"{resolve_preprocessing_report_dir(args.variant) / 'preprocess_manifest.json'}"
    )


if __name__ == "__main__":
    main()
