"""
01_filtrar_dataset.py
Filtra una fuente CSV configurada en config/sources.json para quedarse solo con
filas que tengan valores válidos en ESP y SHIWILU.

Genera trazabilidad de filas removidas y asigna pair_id único por fuente.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "01_filtrado"
REPORTS_DIR = PROJECT_ROOT / "reports" / "01_filtrado"
CONFIG_FILE = PROJECT_ROOT / "config" / "sources.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Nombre de la fuente definida en config/sources.json",
    )
    return parser.parse_args()


def load_sources_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {CONFIG_FILE}")
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_source_config(source_name: str, config: dict) -> dict:
    sources = config.get("sources", [])
    for source in sources:
        if source.get("name") == source_name:
            return source
    raise ValueError(f"Fuente '{source_name}' no encontrada en {CONFIG_FILE}")


def load_raw_data(source_config: dict) -> pd.DataFrame:
    source_path = PROJECT_ROOT / source_config["path"]
    if not source_path.exists():
        raise FileNotFoundError(f"Archivo de fuente no encontrado: {source_path}")

    read_options = source_config.get("read_options", {})
    return pd.read_csv(source_path, encoding="utf-8-sig", **read_options)


def to_work_columns(df: pd.DataFrame, source_config: dict) -> pd.DataFrame:
    esp_col = source_config.get("esp_column", "ESP")
    shi_col = source_config.get("shiwilu_column", "SHIWILU")

    if esp_col not in df.columns or shi_col not in df.columns:
        raise ValueError(
            f"Columnas configuradas no encontradas para {source_config.get('name')}: "
            f"ESP='{esp_col}' SHIWILU='{shi_col}'"
        )

    df_work = pd.DataFrame()
    df_work["ESP"] = df[esp_col]
    df_work["SHIWILU"] = df[shi_col]
    return df_work


def filter_valid_rows(df_work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filtra filas con valores válidos en ESP y SHIWILU.
    
    Retorna:
        - DataFrame con filas válidas
        - DataFrame con filas removidas y motivo
    """
    df_work = df_work.copy()
    df_work["_original_index"] = df_work.index
    
    removed_records = []
    
    # Remove full-empty separator rows silently (common in cotidianas.csv)
    df_work = df_work.dropna(how="all").copy()

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
    
    df_removed = pd.DataFrame(removed_records)
    
    return df_valid, df_removed


def save_outputs(
    df_valid: pd.DataFrame,
    df_removed: pd.DataFrame,
    *,
    source_name: str,
) -> None:
    """Guarda dataset filtrado y log de removidos."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = INTERMEDIATE_DIR / f"{source_name}.csv"
    removed_log_file = REPORTS_DIR / f"{source_name}_rows_removed.csv"
    df_valid.insert(0, "pair_id", [f"P{i:05d}" for i in range(len(df_valid))])
    df_valid.to_csv(output_file, index=False, encoding="utf-8-sig")

    if not df_removed.empty:
        df_removed.to_csv(removed_log_file, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=[
            "original_index", "ESP", "SHIWILU", "removal_reason"
        ]).to_csv(removed_log_file, index=False, encoding="utf-8-sig")
    return output_file, removed_log_file


def print_report(
    *,
    source_name: str,
    input_file: Path,
    output_file: Path,
    removed_log_file: Path,
    total_original: int,
    total_valid: int,
    df_removed: pd.DataFrame
) -> None:
    """Imprime reporte de filtrado."""
    print("=" * 60)
    print("  ETAPA 01: FILTRADO INICIAL")
    print("=" * 60)
    print()
    print(f"  Fuente:               {source_name}")
    print(f"  Entrada:              {input_file}")
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
    print(f"    Dataset:  {output_file}")
    print(f"    Log:      {removed_log_file}")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    config = load_sources_config()
    source_config = get_source_config(args.source, config)
    source_name = source_config["name"]
    source_path = PROJECT_ROOT / source_config["path"]

    df_raw = load_raw_data(source_config)
    total_original = len(df_raw)
    df_work = to_work_columns(df_raw, source_config)
    df_valid, df_removed = filter_valid_rows(df_work)

    output_file, removed_log_file = save_outputs(
        df_valid,
        df_removed,
        source_name=source_name,
    )
    print_report(
        source_name=source_name,
        input_file=source_path,
        output_file=output_file,
        removed_log_file=removed_log_file,
        total_original=total_original,
        total_valid=len(df_valid),
        df_removed=df_removed,
    )


if __name__ == "__main__":
    main()
