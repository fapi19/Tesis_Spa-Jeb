"""Phase 1: canonicalize 04_splits jsonl to bidirectional NMT CSVs.

Reads data/processed/04_splits/{train,valid,test}.jsonl, emits one row per
direction (shw -> spa, spa -> shw) per pair, into
data/processed/05_nmt_canonical/{train,valid,test}.csv.

Schema: id, pair_id, group_id, source, target, source_lang, target_lang,
        split, has_audit_flags, origin_source.

Writes a manifest under reports/05_nmt/preprocessing/canonical_manifest.json
and asserts no group_id leakage across splits.

Invoked from the workspace root as:
    python scripts/nmt/10_canonicalize_dataset.py
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "05_nmt_canonical"
REPORTS_DIR = PROJECT_ROOT / "reports" / "05_nmt" / "preprocessing"

SPLITS = ("train", "valid", "test")

CSV_COLUMNS = [
    "id",
    "pair_id",
    "group_id",
    "source",
    "target",
    "source_lang",
    "target_lang",
    "split",
    "has_audit_flags",
    "origin_source",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _emit_directional(row: dict, split: str) -> list[dict]:
    pair_id = row["pair_id"]
    shw = row["shiwilu"]
    spa = row["spanish"]
    origin_source = row.get("source", "unknown")
    has_flags = bool(row.get("has_audit_flags", False))
    group_id = row["group_id"]

    return [
        {
            "id": f"{pair_id}__shw2spa",
            "pair_id": pair_id,
            "group_id": group_id,
            "source": shw,
            "target": spa,
            "source_lang": "shw",
            "target_lang": "spa",
            "split": split,
            "has_audit_flags": has_flags,
            "origin_source": origin_source,
        },
        {
            "id": f"{pair_id}__spa2shw",
            "pair_id": pair_id,
            "group_id": group_id,
            "source": spa,
            "target": shw,
            "source_lang": "spa",
            "target_lang": "shw",
            "split": split,
            "has_audit_flags": has_flags,
            "origin_source": origin_source,
        },
    ]


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows handles apostrophes/accents correctly;
    # pandas/python csv reads ignore the BOM transparently.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    print(f"[phase1] reading splits from {SPLITS_DIR}")

    per_split_input: dict[str, list[dict]] = {}
    input_sha: dict[str, str] = {}
    for split in SPLITS:
        jsonl_path = SPLITS_DIR / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"[phase1] missing input split: {jsonl_path}", file=sys.stderr)
            return 1
        per_split_input[split] = _read_jsonl(jsonl_path)
        input_sha[split] = _sha256(jsonl_path)
        print(f"[phase1]   {split}: {len(per_split_input[split])} pairs (sha {input_sha[split][:12]})")

    # Group-id leakage check across splits.
    group_to_split: dict[str, str] = {}
    for split, rows in per_split_input.items():
        for row in rows:
            gid = row["group_id"]
            existing = group_to_split.get(gid)
            if existing is not None and existing != split:
                print(
                    f"[phase1] group_id leakage: {gid} appears in both {existing} and {split}",
                    file=sys.stderr,
                )
                return 2
            group_to_split[gid] = split

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_split_output: dict[str, list[dict]] = {}
    output_sha: dict[str, str] = {}
    direction_counts = {"shw2spa": 0, "spa2shw": 0}
    origin_counts: dict[str, int] = {}

    for split, rows in per_split_input.items():
        out_rows: list[dict] = []
        for row in rows:
            for emitted in _emit_directional(row, split):
                out_rows.append(emitted)
                direction_counts[f"{emitted['source_lang']}2{emitted['target_lang']}"] += 1
                origin_counts[emitted["origin_source"]] = (
                    origin_counts.get(emitted["origin_source"], 0) + 1
                )
        out_path = OUT_DIR / f"{split}.csv"
        _write_csv(out_rows, out_path)
        output_sha[split] = _sha256(out_path)
        per_split_output[split] = out_rows
        print(
            f"[phase1] wrote {out_path.relative_to(PROJECT_ROOT)} "
            f"({len(out_rows)} rows, sha {output_sha[split][:12]})"
        )

    total_rows = sum(len(rs) for rs in per_split_output.values())
    expected = 2 * sum(len(rs) for rs in per_split_input.values())
    if total_rows != expected:
        print(
            f"[phase1] total rows mismatch: got {total_rows}, expected {expected}",
            file=sys.stderr,
        )
        return 3

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": 1,
        "step": "canonicalize_dataset",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            split: {
                "path": str((SPLITS_DIR / f"{split}.jsonl").relative_to(PROJECT_ROOT)),
                "sha256": input_sha[split],
                "row_count": len(per_split_input[split]),
            }
            for split in SPLITS
        },
        "outputs": {
            split: {
                "path": str((OUT_DIR / f"{split}.csv").relative_to(PROJECT_ROOT)),
                "sha256": output_sha[split],
                "row_count": len(per_split_output[split]),
            }
            for split in SPLITS
        },
        "direction_counts": direction_counts,
        "origin_counts": origin_counts,
        "total_rows": total_rows,
        "expected_rows": expected,
        "group_id_leakage": False,
        "schema": CSV_COLUMNS,
    }
    manifest_path = REPORTS_DIR / "canonical_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[phase1] manifest -> {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"[phase1] total {total_rows} rows (expected {expected}); group leakage: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
