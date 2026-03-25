"""
01_filtrar_dataset.py
Filtra el CSV original para quedarse solo con filas que tengan
valores válidos en ambas columnas: ESP y SHIWILU.

Genera trazabilidad de filas removidas y asigna pair_id único.

Entrada:  data/raw/flashcards2.csv
Salida:   data/intermediate/01_filtrado/dataset_filtrado.csv
          reports/01_filtrado/rows_removed.csv
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "01_filtrado"
REPORTS_DIR = PROJECT_ROOT / "reports" / "01_filtrado"

INPUT_FILE = RAW_DIR / "flashcards2.csv"
OUTPUT_FILE = INTERMEDIATE_DIR / "dataset_filtrado.csv"
REMOVED_LOG_FILE = REPORTS_DIR / "rows_removed.csv"


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Carga CSV crudo con encoding explícito."""
    return pd.read_csv(filepath, encoding="utf-8")


def filter_valid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtra filas con valores válidos en ESP y SHIWILU.
    
    Retorna:
        - DataFrame con filas válidas
        - DataFrame con filas removidas y motivo
    """
    df_work = df[["ESP", "SHIWILU"]].copy()
    df_work["_original_index"] = df.index
    
    removed_records = []
    
    mask_na_esp = df_work["ESP"].isna()
    mask_na_shi = df_work["SHIWILU"].isna()
    
    for idx in df_work[mask_na_esp].index:
        removed_records.append({
            "original_index": idx,
            "ESP": df_work.loc[idx, "ESP"],
            "SHIWILU": df_work.loc[idx, "SHIWILU"],
            "removal_reason": "ESP es NaN"
        })
    
    for idx in df_work[mask_na_shi & ~mask_na_esp].index:
        removed_records.append({
            "original_index": idx,
            "ESP": df_work.loc[idx, "ESP"],
            "SHIWILU": df_work.loc[idx, "SHIWILU"],
            "removal_reason": "SHIWILU es NaN"
        })
    
    df_work = df_work.dropna(subset=["ESP", "SHIWILU"])
    
    df_work["ESP"] = df_work["ESP"].astype(str)
    df_work["SHIWILU"] = df_work["SHIWILU"].astype(str)
    
    mask_empty_esp = df_work["ESP"].str.strip() == ""
    mask_empty_shi = df_work["SHIWILU"].str.strip() == ""
    mask_placeholder = df_work["SHIWILU"].str.strip() == "--"
    
    for idx in df_work[mask_empty_esp].index:
        removed_records.append({
            "original_index": idx,
            "ESP": df_work.loc[idx, "ESP"],
            "SHIWILU": df_work.loc[idx, "SHIWILU"],
            "removal_reason": "ESP vacío tras strip"
        })
    
    for idx in df_work[mask_empty_shi & ~mask_empty_esp].index:
        removed_records.append({
            "original_index": idx,
            "ESP": df_work.loc[idx, "ESP"],
            "SHIWILU": df_work.loc[idx, "SHIWILU"],
            "removal_reason": "SHIWILU vacío tras strip"
        })
    
    for idx in df_work[mask_placeholder & ~mask_empty_esp & ~mask_empty_shi].index:
        removed_records.append({
            "original_index": idx,
            "ESP": df_work.loc[idx, "ESP"],
            "SHIWILU": df_work.loc[idx, "SHIWILU"],
            "removal_reason": "SHIWILU es placeholder '--'"
        })
    
    mask_invalid = mask_empty_esp | mask_empty_shi | mask_placeholder
    df_valid = df_work[~mask_invalid].copy()
    
    df_valid = df_valid.drop(columns=["_original_index"])
    df_valid = df_valid.reset_index(drop=True)
    
    df_valid.insert(0, "pair_id", [f"P{i:05d}" for i in range(len(df_valid))])
    
    df_removed = pd.DataFrame(removed_records)
    
    return df_valid, df_removed


def save_outputs(
    df_valid: pd.DataFrame,
    df_removed: pd.DataFrame,
    total_original: int
) -> None:
    """Guarda dataset filtrado y log de removidos."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_valid.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    if not df_removed.empty:
        df_removed.to_csv(REMOVED_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=[
            "original_index", "ESP", "SHIWILU", "removal_reason"
        ]).to_csv(REMOVED_LOG_FILE, index=False, encoding="utf-8-sig")


def print_report(
    total_original: int,
    total_valid: int,
    df_removed: pd.DataFrame
) -> None:
    """Imprime reporte de filtrado."""
    print("=" * 60)
    print("  ETAPA 01: FILTRADO INICIAL")
    print("=" * 60)
    print()
    print(f"  Entrada:              {INPUT_FILE}")
    print(f"  Filas originales:     {total_original}")
    print(f"  Filas válidas:        {total_valid}")
    print(f"  Filas removidas:      {len(df_removed)}")
    print()
    
    if not df_removed.empty:
        print("  Detalle de removidas por motivo:")
        for reason, count in df_removed["removal_reason"].value_counts().items():
            print(f"    - {reason}: {count}")
        print()
    
    print(f"  Salidas generadas:")
    print(f"    Dataset:  {OUTPUT_FILE}")
    print(f"    Log:      {REMOVED_LOG_FILE}")
    print("=" * 60)


def main() -> None:
    df_raw = load_raw_data(INPUT_FILE)
    total_original = len(df_raw)
    
    df_valid, df_removed = filter_valid_rows(df_raw)
    
    save_outputs(df_valid, df_removed, total_original)
    print_report(total_original, len(df_valid), df_removed)


if __name__ == "__main__":
    main()
