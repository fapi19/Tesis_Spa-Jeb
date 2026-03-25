"""
00_extraer_dataset_pdf.py
Extrae pares Shiwilu-Castellano desde un PDF con estructura numerada.

Entrada:  data/raw/II_TEXTOS_SHIWILU.pdf
Salidas:  data/intermediate/00_pdf/dataset_extraido_pdf.csv
          reports/00_pdf/summary.json

Nota metodologica:
- El parser usa heuristicas conservadoras para separar SHIWILU y ESP.
- No elimina informacion linguistica; solo limpia ruido estructural del PDF.
- Marca casos ambiguos para revision manual.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "00_pdf"
REPORTS_DIR = PROJECT_ROOT / "reports" / "00_pdf"

INPUT_PDF = RAW_DIR / "II_TEXTOS_SHIWILU.pdf"
OUTPUT_CSV = INTERMEDIATE_DIR / "dataset_extraido_pdf.csv"
SUMMARY_JSON = REPORTS_DIR / "summary.json"

ITEM_PATTERN = re.compile(r"^(\d+)\.\s*(.+)$")
PAGE_MARKER_PATTERN = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--$")
FOOTNOTE_PATTERN = re.compile(r"^\d+\s+.+$")
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")

SPANISH_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "y",
    "que",
    "no",
    "me",
    "mi",
    "con",
    "por",
    "para",
    "aqui",
    "ahora",
    "como",
    "porque",
    "una",
    "un",
    "en",
    "se",
    "yo",
}


@dataclass
class NumberedItem:
    texto_id: int | None
    item_number: int
    raw_lines: list[str]


def normalize_spaces(text: str) -> str:
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def read_pdf_lines(pdf_path: Path) -> list[str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se encontro el PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    lines: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        lines.extend(page_text.splitlines())
    return [line.strip() for line in lines]


def is_skippable_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_MARKER_PATTERN.match(line):
        return True
    if line.startswith("TEXTO "):
        return True
    if re.match(r"^[A-Za-zÁÉÍÓÚáéíóúÑñ].*,\s*\d{4}$", line):
        return True
    return False


def collect_numbered_items(lines: list[str]) -> list[NumberedItem]:
    items: list[NumberedItem] = []
    current_texto_id: int | None = None
    current_number: int | None = None
    current_lines: list[str] = []

    for line in lines:
        text_match = re.match(r"^TEXTO\s+(\d+)$", line)
        if text_match:
            if current_number is not None:
                items.append(
                    NumberedItem(
                        texto_id=current_texto_id,
                        item_number=current_number,
                        raw_lines=current_lines,
                    )
                )
                current_number = None
                current_lines = []
            current_texto_id = int(text_match.group(1))
            continue

        if is_skippable_line(line):
            continue

        match = ITEM_PATTERN.match(line)
        if match:
            if current_number is not None:
                items.append(
                    NumberedItem(
                        texto_id=current_texto_id,
                        item_number=current_number,
                        raw_lines=current_lines,
                    )
                )

            current_number = int(match.group(1))
            first_content = normalize_spaces(match.group(2))
            current_lines = [first_content] if first_content else []
            continue

        if current_number is not None:
            # Filtra notas de pie de pagina numeradas del tipo:
            # "1 U'chimu para otros hablantes."
            if FOOTNOTE_PATTERN.match(line) and "para otros hablantes" in line.lower():
                continue
            current_lines.append(normalize_spaces(line))

    if current_number is not None:
        items.append(
            NumberedItem(
                texto_id=current_texto_id,
                item_number=current_number,
                raw_lines=current_lines,
            )
        )

    return items


def spanish_score(line: str) -> int:
    lowered = re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+", " ", line.lower())
    tokens = [t for t in lowered.split() if t]
    stopword_hits = sum(1 for token in tokens if token in SPANISH_STOPWORDS)
    accent_hint = int(bool(re.search(r"[áéíóúÁÉÍÓÚ¿¡]", line)))
    apostrophe_penalty = line.count("'") + line.count("’")
    return stopword_hits + accent_hint - apostrophe_penalty


def shiwilu_score(line: str) -> int:
    apostrophes = line.count("'") + line.count("’")
    clusters = len(re.findall(r"(sh|ku|na|pi|llu|wek|ñi)", line.lower()))
    return apostrophes + clusters


def split_bilingual_lines(raw_lines: list[str]) -> tuple[str, str, str]:
    """
    Separa un bloque numerado en SHIWILU y ESP con heuristicas:
    - La primera linea se asume SHIWILU.
    - El cambio a ESP ocurre cuando el score de espanol supera al de shiwilu.
    """
    if not raw_lines:
        return "", "", "empty_block"

    shiwilu_lines: list[str] = []
    esp_lines: list[str] = []
    mode = "shiwilu"

    for i, line in enumerate(raw_lines):
        if not line:
            continue

        if i == 0:
            shiwilu_lines.append(line)
            continue

        s_score = spanish_score(line)
        sh_score = shiwilu_score(line)

        if mode == "shiwilu":
            if s_score > sh_score:
                mode = "esp"
                esp_lines.append(line)
            else:
                shiwilu_lines.append(line)
        else:
            esp_lines.append(line)

    notes: list[str] = []

    if not esp_lines and len(shiwilu_lines) > 1:
        # fallback: ultima linea como espanol si no se detecto cambio
        esp_lines = [shiwilu_lines.pop()]
        notes.append("fallback_last_line_as_spanish")

    shiwilu = normalize_spaces(" ".join(shiwilu_lines))
    esp = normalize_spaces(" ".join(esp_lines))

    if not shiwilu:
        notes.append("missing_shiwilu")
    if not esp:
        notes.append("missing_spanish")

    return shiwilu, esp, "|".join(notes) if notes else "ok"


def build_parallel_dataframe(items: list[NumberedItem]) -> pd.DataFrame:
    records: list[dict[str, str | int]] = []
    for item in items:
        shiwilu, esp, quality_flag = split_bilingual_lines(item.raw_lines)
        records.append(
            {
                "texto_id": item.texto_id if item.texto_id is not None else -1,
                "item_number": item.item_number,
                "SHIWILU": shiwilu,
                "ESP": esp,
                "quality_flag": quality_flag,
            }
        )

    df = pd.DataFrame(records).sort_values(["texto_id", "item_number"]).reset_index(drop=True)
    df.insert(
        0,
        "pair_id",
        [
            f"T{int(row.texto_id):03d}_N{int(row.item_number):03d}"
            for row in df.itertuples(index=False)
        ],
    )
    return df


def save_outputs(df: pd.DataFrame, total_items: int) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "pipeline": "00_extraer_dataset_pdf",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_pdf": str(INPUT_PDF),
        "output_csv": str(OUTPUT_CSV),
        "total_numbered_items": total_items,
        "total_rows_exported": int(len(df)),
        "rows_with_ok_flag": int((df["quality_flag"] == "ok").sum()),
        "rows_with_flags": int((df["quality_flag"] != "ok").sum()),
        "quality_flag_counts": df["quality_flag"].value_counts().to_dict(),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(df: pd.DataFrame, total_items: int) -> None:
    print("=" * 70)
    print("  EXTRACCION PDF SHIWILU-CASTELLANO")
    print("=" * 70)
    print()
    print(f"  PDF de entrada:             {INPUT_PDF}")
    print(f"  Items numerados detectados: {total_items}")
    print(f"  Filas exportadas:           {len(df)}")
    print(f"  Filas OK:                   {(df['quality_flag'] == 'ok').sum()}")
    print(f"  Filas con flags:            {(df['quality_flag'] != 'ok').sum()}")
    print()
    print("  Flags de calidad:")
    for flag, count in df["quality_flag"].value_counts().items():
        print(f"    - {flag}: {count}")
    print()
    print("  Salidas:")
    print(f"    CSV:      {OUTPUT_CSV}")
    print(f"    Resumen:  {SUMMARY_JSON}")
    print("=" * 70)


def main() -> None:
    lines = read_pdf_lines(INPUT_PDF)
    items = collect_numbered_items(lines)
    df = build_parallel_dataframe(items)
    save_outputs(df, total_items=len(items))
    print_report(df, total_items=len(items))


if __name__ == "__main__":
    main()
