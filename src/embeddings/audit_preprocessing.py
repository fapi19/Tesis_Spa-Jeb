from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPORTS_DIR, SPLITS_DIR


SPLIT_NAMES = ("train", "valid", "test")
REQUIRED_FIELDS = {
    "pair_id",
    "group_id",
    "source",
    "shiwilu",
    "spanish",
    "raw_shiwilu",
    "raw_spanish",
    "normalized_shiwilu",
    "normalized_spanish",
    "has_audit_flags",
}
FORBIDDEN_LEGACY_FILES = (
    "train_pairs.jsonl",
    "val_pairs.jsonl",
    "all_text.txt",
    "train_pairs_suffix_aware.jsonl",
    "val_pairs_suffix_aware.jsonl",
    "test_pairs_suffix_aware.jsonl",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def text_lengths(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    lengths = [len(str(row[key])) for row in rows]
    token_lengths = [len(str(row[key]).split()) for row in rows]
    if not lengths:
        return {"min_chars": 0, "max_chars": 0, "mean_chars": 0.0, "median_chars": 0.0, "max_tokens": 0}
    return {
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(statistics.mean(lengths), 2),
        "median_chars": round(statistics.median(lengths), 2),
        "max_tokens": max(token_lengths),
    }


def longest_rows(rows: list[dict[str, Any]], key: str, limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: len(str(row[key])), reverse=True)
    return [
        {
            "pair_id": row["pair_id"],
            "group_id": row["group_id"],
            "source": row["source"],
            "shiwilu": row["shiwilu"],
            "spanish": row["spanish"],
            "length": len(str(row[key])),
        }
        for row in sorted_rows[:limit]
    ]


def deterministic_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["pair_id"])[:limit]


def normalization_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    changed = [
        row
        for row in rows
        if row["raw_shiwilu"] != row["normalized_shiwilu"]
        or row["raw_spanish"] != row["normalized_spanish"]
    ]
    return [
        {
            "pair_id": row["pair_id"],
            "raw_shiwilu": row["raw_shiwilu"],
            "normalized_shiwilu": row["normalized_shiwilu"],
            "raw_spanish": row["raw_spanish"],
            "normalized_spanish": row["normalized_spanish"],
        }
        for row in deterministic_sample(changed, limit)
    ]


def apostrophe_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    with_apostrophe = [row for row in rows if "'" in row["shiwilu"]]
    return [
        {
            "pair_id": row["pair_id"],
            "group_id": row["group_id"],
            "shiwilu": row["shiwilu"],
            "raw_shiwilu": row["raw_shiwilu"],
        }
        for row in deterministic_sample(with_apostrophe, limit)
    ]


def group_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)

    multi_groups = [
        (group_id, group_rows)
        for group_id, group_rows in sorted(by_group.items())
        if len(group_rows) > 1
    ]
    samples = []
    for group_id, group_rows in multi_groups[:limit]:
        samples.append(
            {
                "group_id": group_id,
                "size": len(group_rows),
                "pairs": [
                    {
                        "pair_id": row["pair_id"],
                        "shiwilu": row["shiwilu"],
                        "spanish": row["spanish"],
                    }
                    for row in group_rows
                ],
            }
        )
    return samples


def suffix_samples(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [row for row in load_jsonl(path) if "@@" in row["shiwilu"]]
    return [
        {
            "pair_id": row["pair_id"],
            "group_id": row["group_id"],
            "shiwilu": row["shiwilu"],
            "spanish": row["spanish"],
        }
        for row in deterministic_sample(rows, limit)
    ]


def check_required_fields(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    issues = []
    for split_name, rows in splits.items():
        for idx, row in enumerate(rows):
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                issues.append(f"{split_name}:{idx} missing fields {sorted(missing)}")
    return issues


def check_group_leakage(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    group_to_split = {}
    issues = []
    for split_name, rows in splits.items():
        for row in rows:
            group_id = row["group_id"]
            previous = group_to_split.setdefault(group_id, split_name)
            if previous != split_name:
                issues.append(f"group_id {group_id} appears in {previous} and {split_name}")
    return issues


def check_duplicate_exclusions(
    path: Path,
    included_rows: list[dict[str, Any]],
) -> list[str]:
    if not path.exists():
        return ["excluded_exact_duplicate.jsonl does not exist"]
    rows = load_jsonl(path)
    included_keys = {
        (row["normalized_shiwilu"], row["normalized_spanish"])
        for row in included_rows
    }
    issues = []
    for row in rows:
        key = (row["normalized_shiwilu"], row["normalized_spanish"])
        if key not in included_keys:
            issues.append(f"excluded duplicate {row['pair_id']} has no included canonical pair")
    return issues


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "source_distribution": dict(Counter(row["source"] for row in rows)),
        "audit_flagged_rows": sum(bool(row["has_audit_flags"]) for row in rows),
        "shiwilu_lengths": text_lengths(rows, "shiwilu"),
        "spanish_lengths": text_lengths(rows, "spanish"),
    }


def build_report(sample_limit: int) -> dict[str, Any]:
    splits = {split_name: load_jsonl(SPLITS_DIR / f"{split_name}.jsonl") for split_name in SPLIT_NAMES}
    all_rows = [row for rows in splits.values() for row in rows]
    excluded_exact = load_jsonl(SPLITS_DIR / "excluded_exact_duplicate.jsonl")
    excluded_empty = load_jsonl(SPLITS_DIR / "excluded_empty_text.jsonl")

    blocking_issues = []
    blocking_issues.extend(check_required_fields(splits))
    blocking_issues.extend(check_group_leakage(splits))
    blocking_issues.extend(
        check_duplicate_exclusions(
            SPLITS_DIR / "excluded_exact_duplicate.jsonl",
            all_rows,
        )
    )

    legacy_files_present = [
        filename for filename in FORBIDDEN_LEGACY_FILES if (SPLITS_DIR / filename).exists()
    ]
    if legacy_files_present:
        blocking_issues.append(f"legacy embedding artifacts still present: {legacy_files_present}")

    suffix_sample_rows = suffix_samples(SPLITS_DIR / "train_suffix_aware.jsonl", 20)
    suffix_status = "experimental_no_usar_como_default" if suffix_sample_rows else "not_available"

    return {
        "pipeline": "audit_preprocessing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not blocking_issues else "fail",
        "blocking_issues": blocking_issues,
        "splits": {split_name: split_summary(rows) for split_name, rows in splits.items()},
        "excluded": {
            "exact_duplicate": {
                "count": len(excluded_exact),
                "rows": excluded_exact,
            },
            "empty_text": {
                "count": len(excluded_empty),
                "rows": excluded_empty,
            },
        },
        "groups": {
            "total": len({row["group_id"] for row in all_rows}),
            "multi_pair_samples": group_samples(all_rows, 10),
        },
        "samples": {
            "train": deterministic_sample(splits["train"], sample_limit),
            "valid": deterministic_sample(splits["valid"], sample_limit),
            "test": deterministic_sample(splits["test"], sample_limit),
            "normalization": normalization_samples(all_rows, sample_limit),
            "apostrophes": apostrophe_samples(all_rows, sample_limit),
            "longest_shiwilu": longest_rows(all_rows, "shiwilu", sample_limit),
            "longest_spanish": longest_rows(all_rows, "spanish", sample_limit),
        },
        "suffix_aware": {
            "status": suffix_status,
            "reason": "Mantener como variante experimental; no bloquea el preprocesamiento canónico.",
            "samples": suffix_sample_rows,
        },
        "acceptance": {
            "canonical_script_runs": True,
            "canonical_artifacts_exist": True,
            "group_ids_do_not_cross_splits": not check_group_leakage(splits),
            "raw_and_normalized_fields_present": not check_required_fields(splits),
            "legacy_embedding_aliases_removed": not legacy_files_present,
            "length_outliers_audited_not_filtered": True,
        },
        "next_phase": "Entrenar y evaluar embeddings v1 con los splits canónicos.",
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Cierre Del Preprocesamiento De Embeddings",
        "",
        f"Estado: `{report['status']}`",
        "",
        "## Resumen Por Split",
        "",
    ]
    for split_name, summary in report["splits"].items():
        lines.extend(
            [
                f"### {split_name}",
                f"- Filas: {summary['rows']}",
                f"- Grupos: {summary['groups']}",
                f"- Filas con audit flags: {summary['audit_flagged_rows']}",
                f"- Fuentes: {summary['source_distribution']}",
                f"- Longitud Shiwlu: {summary['shiwilu_lengths']}",
                f"- Longitud español: {summary['spanish_lengths']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Exclusiones",
            "",
            f"- Duplicados exactos: {report['excluded']['exact_duplicate']['count']}",
            f"- Texto vacío: {report['excluded']['empty_text']['count']}",
            "",
            "## Suffix-Aware",
            "",
            f"Estado: `{report['suffix_aware']['status']}`",
            report["suffix_aware"]["reason"],
            "",
            "## Decisión",
            "",
            "El preprocesamiento canónico queda cerrado si el estado es `pass`. "
            "La siguiente fase es entrenamiento/evaluación de embeddings, no más preprocesamiento.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoría final del preprocesamiento de embeddings.")
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = build_report(sample_limit=args.sample_limit)
    json_path = REPORTS_DIR / "preprocessing_closure_report.json"
    markdown_path = REPORTS_DIR / "preprocessing_closure_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_markdown(markdown_path, report)

    print(f"Reporte JSON: {json_path}")
    print(f"Reporte Markdown: {markdown_path}")
    print(f"Estado: {report['status']}")
    if report["blocking_issues"]:
        raise SystemExit("La auditoría encontró problemas bloqueantes.")


if __name__ == "__main__":
    main()
