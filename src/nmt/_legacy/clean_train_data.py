from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path


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


def load_pairs(path: Path) -> list[dict[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def write_jsonl(pairs: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def clean(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    backup_path = input_path.parent / "train_pairs_original.jsonl"
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"Backup creado: {backup_path}")

    raw_pairs = load_pairs(input_path)
    total_original = len(raw_pairs)
    print(f"Pares cargados: {total_original}")

    # --- Normalize -----------------------------------------------------------
    normalized = []
    for p in raw_pairs:
        normalized.append({
            "shiwilu": normalize_text(p["shiwilu"]),
            "spanish": normalize_text(p["spanish"]),
        })

    # --- Editorial / glosas --------------------------------------------------
    editorial: list[dict[str, str]] = []
    non_editorial: list[dict[str, str]] = []
    for p in normalized:
        if has_editorial_markers(p["shiwilu"], p["spanish"]):
            editorial.append(p)
        else:
            non_editorial.append(p)
    print(f"Editorial/glosas detectados: {len(editorial)}")

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
    for shw, variants in by_shw.items():
        unique_spa = {v["spanish"] for v in variants}
        if len(unique_spa) >= 2:
            conflicts.extend(variants)
            canonical = min(unique_spa, key=len)
            no_conflict.append({"shiwilu": shw, "spanish": canonical})
        else:
            no_conflict.append(variants[0])
    print(f"Fuentes con conflicto (>1 trad): {len(by_shw) - len(no_conflict) + sum(1 for s, vs in by_shw.items() if len({v['spanish'] for v in vs}) >= 2)}")

    # --- Length filter -------------------------------------------------------
    narrative: list[dict[str, str]] = []
    too_short: list[dict[str, str]] = []
    imbalanced: list[dict[str, str]] = []
    clean_pairs: list[dict[str, str]] = []

    nt = args.narrative_threshold
    rt = args.ratio_threshold

    for p in no_conflict:
        wc_shw = word_count(p["shiwilu"])
        wc_spa = word_count(p["spanish"])

        if wc_shw == 0 or wc_spa == 0:
            too_short.append(p)
            continue

        if wc_shw > nt or wc_spa > (nt + 5):
            narrative.append(p)
            continue

        if length_ratio(p["shiwilu"], p["spanish"]) > rt:
            editorial.append(p)
            imbalanced.append(p)
            continue

        clean_pairs.append(p)

    print(f"Narrativa larga: {len(narrative)}")
    print(f"Desbalance extremo (-> editorial): {len(imbalanced)}")
    print(f"Pares vacios/1-token descartados: {len(too_short)}")
    print(f"train_clean final: {len(clean_pairs)}")

    # --- Write outputs -------------------------------------------------------
    write_jsonl(clean_pairs, output_dir / "train_clean.jsonl")
    write_jsonl(conflicts, output_dir / "train_conflicts.jsonl")
    write_jsonl(narrative, output_dir / "train_narrative.jsonl")
    write_jsonl(editorial, output_dir / "train_editorial.jsonl")

    report = {
        "total_original": total_original,
        "duplicates_removed": n_dupes,
        "editorial_glosas": len(editorial),
        "conflicts_sources": len([
            s for s, vs in by_shw.items()
            if len({v["spanish"] for v in vs}) >= 2
        ]),
        "conflicts_pairs": len(conflicts),
        "narrative": len(narrative),
        "imbalanced_to_editorial": len(imbalanced),
        "too_short_discarded": len(too_short),
        "train_clean": len(clean_pairs),
        "thresholds": {
            "narrative_tokens": nt,
            "ratio": rt,
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReporte guardado: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Limpieza y segmentacion de datos de entrenamiento NMT"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al archivo train_pairs.jsonl original",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde escribir los 4 JSONL de salida",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/nmt",
        help="Directorio para el reporte JSON (default: reports/nmt)",
    )
    parser.add_argument(
        "--narrative-threshold",
        type=int,
        default=25,
        help="Tokens src para considerar narrativa (default: 25)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=4.0,
        help="Umbral de desbalance src/tgt (default: 4.0)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    clean(args)


if __name__ == "__main__":
    main()
