"""
01b_unificar_fuentes.py
Unifica múltiples fuentes de datos en un solo dataset para el pipeline.

Lee las fuentes configuradas en config/sources.json y las combina en un
único dataset con trazabilidad de origen.

Entrada:  config/sources.json (lista de fuentes)
          data/intermediate/01_filtrado/dataset_filtrado.csv (flashcards)
          data/intermediate/00_pdf/dataset_extraido_pdf.csv (PDF)

Salida:   data/intermediate/01b_unificado/dataset_unificado.csv
          reports/01b_unificado/summary.json
          reports/01b_unificado/cross_duplicates.csv
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "sources.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate" / "01b_unificado"
REPORTS_DIR = PROJECT_ROOT / "reports" / "01b_unificado"

OUTPUT_FILE = OUTPUT_DIR / "dataset_unificado.csv"
SUMMARY_FILE = REPORTS_DIR / "summary.json"
CROSS_DUPLICATES_FILE = REPORTS_DIR / "cross_duplicates.csv"


def load_sources_config() -> dict[str, Any]:
    """Carga configuración de fuentes desde JSON."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source(source_config: dict) -> pd.DataFrame | None:
    """
    Carga una fuente individual según su configuración.
    
    Retorna None si la fuente está deshabilitada o el archivo no existe.
    """
    if not source_config.get("enabled", False):
        return None
    
    source_path = PROJECT_ROOT / source_config["path"]
    if not source_path.exists():
        print(f"  ADVERTENCIA: Archivo no encontrado para fuente '{source_config['name']}': {source_path}")
        return None
    
    df = pd.read_csv(source_path, encoding="utf-8-sig")
    
    esp_col = source_config.get("esp_column", "ESP")
    shi_col = source_config.get("shiwilu_column", "SHIWILU")
    pair_id_col = source_config.get("pair_id_column", "pair_id")
    
    if esp_col not in df.columns or shi_col not in df.columns:
        print(f"  ADVERTENCIA: Columnas ESP/SHIWILU no encontradas en '{source_config['name']}'")
        return None
    
    df_out = pd.DataFrame()
    df_out["ESP"] = df[esp_col].astype(str)
    df_out["SHIWILU"] = df[shi_col].astype(str)
    df_out["source"] = source_config["name"]
    
    if pair_id_col in df.columns:
        df_out["source_pair_id"] = df[pair_id_col].astype(str)
    else:
        df_out["source_pair_id"] = [f"{source_config['name']}_{i}" for i in range(len(df))]
    
    filter_config = source_config.get("filter")
    if filter_config:
        filter_col = filter_config.get("column")
        keep_values = filter_config.get("keep", [])
        
        if filter_col and filter_col in df.columns and keep_values:
            mask = df[filter_col].isin(keep_values)
            df_out = df_out[mask.values].copy()
    
    return df_out


def detect_cross_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta pares ESP+SHIWILU que aparecen en más de una fuente.
    
    Retorna DataFrame con los duplicados cross-source.
    """
    df_lower = df.copy()
    df_lower["esp_lower"] = df_lower["ESP"].str.lower().str.strip()
    df_lower["shi_lower"] = df_lower["SHIWILU"].str.lower().str.strip()
    
    grouped = df_lower.groupby(["esp_lower", "shi_lower"])["source"].nunique()
    cross_dups = grouped[grouped > 1]
    
    if cross_dups.empty:
        return pd.DataFrame(columns=[
            "pair_id", "ESP", "SHIWILU", "source", "source_pair_id", "duplicate_group"
        ])
    
    dup_keys = set(cross_dups.index)
    mask = df_lower.apply(
        lambda row: (row["esp_lower"], row["shi_lower"]) in dup_keys,
        axis=1
    )
    
    df_dups = df[mask].copy()
    
    df_dups["duplicate_group"] = df_lower.loc[mask].apply(
        lambda row: f"{row['esp_lower'][:20]}|{row['shi_lower'][:20]}",
        axis=1
    )
    
    return df_dups.sort_values(["duplicate_group", "source"]).reset_index(drop=True)


def unify_sources(config: dict) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """
    Unifica todas las fuentes habilitadas en un solo DataFrame.
    
    Returns:
        - DataFrame unificado
        - Diccionario con conteos por fuente
        - DataFrame de duplicados cross-source
    """
    sources = config.get("sources", [])
    dfs: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    
    for source_config in sources:
        name = source_config.get("name", "unknown")
        print(f"  Cargando fuente: {name}...")
        
        df = load_source(source_config)
        if df is not None and not df.empty:
            counts[name] = len(df)
            dfs.append(df)
            print(f"    -> {len(df)} filas cargadas")
        else:
            counts[name] = 0
            print(f"    -> 0 filas (deshabilitada o vacía)")
    
    if not dfs:
        return pd.DataFrame(columns=["pair_id", "ESP", "SHIWILU", "source", "source_pair_id"]), counts, pd.DataFrame()
    
    df_unified = pd.concat(dfs, ignore_index=True)
    
    df_unified.insert(0, "pair_id", [f"U{i:05d}" for i in range(len(df_unified))])
    
    cross_dups = detect_cross_duplicates(df_unified)
    
    return df_unified, counts, cross_dups


def save_outputs(
    df: pd.DataFrame,
    counts: dict[str, int],
    cross_dups: pd.DataFrame
) -> None:
    """Guarda todas las salidas del script."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    cross_dups.to_csv(CROSS_DUPLICATES_FILE, index=False, encoding="utf-8-sig")
    
    summary = {
        "pipeline": "01b_unificar_fuentes",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_file": str(CONFIG_FILE),
        "output_file": str(OUTPUT_FILE),
        "sources": counts,
        "total_rows": len(df),
        "cross_source_duplicates": len(cross_dups),
        "unique_sources": len([k for k, v in counts.items() if v > 0])
    }
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(df: pd.DataFrame, counts: dict[str, int], cross_dups: pd.DataFrame) -> None:
    """Imprime reporte de unificación en consola."""
    print("=" * 70)
    print("  ETAPA 01b: UNIFICACIÓN DE FUENTES")
    print("=" * 70)
    print()
    print(f"  Configuración:            {CONFIG_FILE}")
    print()
    print("  FUENTES PROCESADAS:")
    print("  " + "-" * 50)
    for source, count in counts.items():
        status = "[OK]" if count > 0 else "[--]"
        print(f"    {status} {source}: {count} filas")
    print()
    print(f"  Total filas unificadas:   {len(df)}")
    print(f"  Duplicados cross-source:  {len(cross_dups)}")
    print()
    
    if not cross_dups.empty:
        print("  DUPLICADOS ENTRE FUENTES (primeros 5):")
        print("  " + "-" * 50)
        for _, row in cross_dups.head(5).iterrows():
            print(f"    [{row['source']}] {row['ESP'][:30]}... <-> {row['SHIWILU'][:30]}...")
        if len(cross_dups) > 5:
            print(f"    ... y {len(cross_dups) - 5} más")
        print()
    
    print("  SALIDAS GENERADAS:")
    print("  " + "-" * 50)
    print(f"    Dataset:     {OUTPUT_FILE}")
    print(f"    Resumen:     {SUMMARY_FILE}")
    print(f"    Duplicados:  {CROSS_DUPLICATES_FILE}")
    print("=" * 70)


def main() -> None:
    print()
    config = load_sources_config()
    df, counts, cross_dups = unify_sources(config)
    save_outputs(df, counts, cross_dups)
    print_report(df, counts, cross_dups)


if __name__ == "__main__":
    main()
