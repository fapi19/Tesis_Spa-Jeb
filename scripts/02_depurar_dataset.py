"""
02_depurar_dataset.py
Pipeline de normalización no destructiva del corpus español-shiwilu.

Mantiene columnas originales y crea versiones normalizadas.
Registra todas las transformaciones aplicadas en bitácora.

Entrada:  data/intermediate/01b_unificado/dataset_unificado.csv
          config/normalization_rules.json

Salida:   data/intermediate/02_normalizado/dataset_normalizado.csv
          reports/02_normalizacion/normalization_log.csv
          reports/02_normalizacion/rows_removed.csv
          reports/02_normalizacion/summary.json
"""

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "intermediate" / "01b_unificado"
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate" / "02_normalizado"
REPORTS_DIR = PROJECT_ROOT / "reports" / "02_normalizacion"
CONFIG_DIR = PROJECT_ROOT / "config"

INPUT_FILE = INPUT_DIR / "dataset_unificado.csv"
OUTPUT_FILE = OUTPUT_DIR / "dataset_normalizado.csv"
CONFIG_FILE = CONFIG_DIR / "normalization_rules.json"

NORMALIZATION_LOG_FILE = REPORTS_DIR / "normalization_log.csv"
REMOVED_LOG_FILE = REPORTS_DIR / "rows_removed.csv"
SUMMARY_FILE = REPORTS_DIR / "summary.json"

MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")


def load_config() -> dict[str, Any]:
    """Carga configuración de reglas de normalización."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> pd.DataFrame:
    """Carga dataset filtrado."""
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    return df


def apply_unicode_nfc(text: str) -> str:
    """Normalización Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def apply_trim(text: str) -> str:
    """Eliminar espacios al inicio y final."""
    return text.strip()


def apply_collapse_spaces(text: str) -> str:
    """Colapsar múltiples espacios en uno."""
    return MULTI_SPACE_PATTERN.sub(" ", text)


def apply_normalize_comma_space(text: str) -> str:
    """Normalizar ' , ' a ', '."""
    return text.replace(" , ", ", ")


def apply_lowercase(text: str) -> str:
    """Convertir a minúsculas."""
    return text.lower()


RULE_FUNCTIONS = {
    "unicode_nfc": apply_unicode_nfc,
    "trim": apply_trim,
    "collapse_spaces": apply_collapse_spaces,
    "normalize_comma_space": apply_normalize_comma_space,
    "lowercase": apply_lowercase,
}


def get_active_global_rules(config: dict) -> list[tuple[str, callable]]:
    """
    Obtiene lista ordenada de reglas globales activas.
    
    Retorna lista de tuplas (nombre_regla, función).
    """
    global_rules = config.get("global_rules", {})
    active_rules = []
    
    for rule_name, rule_config in global_rules.items():
        if rule_config.get("enabled", False):
            order = rule_config.get("order", 999)
            if rule_name in RULE_FUNCTIONS:
                active_rules.append((order, rule_name, RULE_FUNCTIONS[rule_name]))
    
    active_rules.sort(key=lambda x: x[0])
    return [(name, func) for _, name, func in active_rules]


def normalize_text(
    text: str,
    rules: list[tuple[str, callable]],
    pair_id: str,
    column: str,
    log_records: list[dict]
) -> str:
    """
    Aplica reglas de normalización secuencialmente y registra cambios.
    
    Args:
        text: Texto original
        rules: Lista de (nombre_regla, función)
        pair_id: ID del par para trazabilidad
        column: Nombre de columna (ESP o SHIWILU)
        log_records: Lista donde agregar registros de cambios
    
    Returns:
        Texto normalizado
    """
    current_text = text
    
    for rule_name, rule_func in rules:
        new_text = rule_func(current_text)
        
        if new_text != current_text:
            log_records.append({
                "pair_id": pair_id,
                "column": column,
                "rule_name": rule_name,
                "before": current_text,
                "after": new_text
            })
        
        current_text = new_text
    
    return current_text


def process_dataset(
    df: pd.DataFrame,
    config: dict
) -> tuple[pd.DataFrame, list[dict], list[dict], dict]:
    """
    Procesa el dataset aplicando normalización no destructiva.
    
    Returns:
        - DataFrame con columnas originales y normalizadas
        - Lista de registros de normalización
        - Lista de filas removidas (vacías por defecto en esta etapa)
        - Diccionario de estadísticas
    """
    active_rules = get_active_global_rules(config)
    
    df_out = df.copy()
    df_out["ESP_original"] = df_out["ESP"].astype(str)
    df_out["SHIWILU_original"] = df_out["SHIWILU"].astype(str)
    
    log_records: list[dict] = []
    removed_records: list[dict] = []
    
    esp_normalized = []
    shi_normalized = []
    
    for idx, row in df_out.iterrows():
        pair_id = row["pair_id"]
        
        esp_norm = normalize_text(
            row["ESP_original"],
            active_rules,
            pair_id,
            "ESP",
            log_records
        )
        esp_normalized.append(esp_norm)
        
        shi_norm = normalize_text(
            row["SHIWILU_original"],
            active_rules,
            pair_id,
            "SHIWILU",
            log_records
        )
        shi_normalized.append(shi_norm)
    
    df_out["ESP_normalizado"] = esp_normalized
    df_out["SHIWILU_normalizado"] = shi_normalized
    
    output_columns = [
        "pair_id",
        "ESP_original",
        "SHIWILU_original",
        "ESP_normalizado",
        "SHIWILU_normalizado"
    ]
    
    if "source" in df_out.columns:
        output_columns.append("source")
    if "source_pair_id" in df_out.columns:
        output_columns.append("source_pair_id")
    
    df_out = df_out[output_columns]
    
    stats = {
        "total_rows": len(df_out),
        "rules_applied": [r[0] for r in active_rules],
        "rows_with_changes": len(set(r["pair_id"] for r in log_records)),
        "total_transformations": len(log_records),
        "transformations_by_rule": {},
        "transformations_by_column": {"ESP": 0, "SHIWILU": 0}
    }
    
    for record in log_records:
        rule = record["rule_name"]
        col = record["column"]
        stats["transformations_by_rule"][rule] = stats["transformations_by_rule"].get(rule, 0) + 1
        stats["transformations_by_column"][col] += 1
    
    return df_out, log_records, removed_records, stats


def save_outputs(
    df: pd.DataFrame,
    log_records: list[dict],
    removed_records: list[dict],
    stats: dict,
    config: dict
) -> None:
    """Guarda todas las salidas del pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    if log_records:
        df_log = pd.DataFrame(log_records)
        df_log.to_csv(NORMALIZATION_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=[
            "pair_id", "column", "rule_name", "before", "after"
        ]).to_csv(NORMALIZATION_LOG_FILE, index=False, encoding="utf-8-sig")
    
    if removed_records:
        df_removed = pd.DataFrame(removed_records)
        df_removed.to_csv(REMOVED_LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=[
            "pair_id", "ESP_original", "SHIWILU_original", "removal_reason"
        ]).to_csv(REMOVED_LOG_FILE, index=False, encoding="utf-8-sig")
    
    summary = {
        "pipeline": "02_depurar_dataset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "config_file": str(CONFIG_FILE),
        "statistics": stats,
        "config_snapshot": {
            "global_rules_enabled": [
                name for name, cfg in config.get("global_rules", {}).items()
                if cfg.get("enabled", False)
            ],
            "language_specific_rules_enabled": {
                lang: [
                    name for name, cfg in rules.items()
                    if cfg.get("enabled", False)
                ]
                for lang, rules in config.get("language_specific", {}).items()
            }
        }
    }
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(stats: dict) -> None:
    """Imprime reporte de normalización."""
    print("=" * 60)
    print("  ETAPA 02: NORMALIZACIÓN NO DESTRUCTIVA")
    print("=" * 60)
    print()
    print(f"  Entrada:                    {INPUT_FILE}")
    print(f"  Configuración:              {CONFIG_FILE}")
    print()
    print(f"  Filas procesadas:           {stats['total_rows']}")
    print(f"  Filas con cambios:          {stats['rows_with_changes']}")
    print(f"  Total transformaciones:     {stats['total_transformations']}")
    print()
    print("  Reglas aplicadas (en orden):")
    for rule in stats["rules_applied"]:
        count = stats["transformations_by_rule"].get(rule, 0)
        print(f"    - {rule}: {count} cambios")
    print()
    print("  Transformaciones por columna:")
    for col, count in stats["transformations_by_column"].items():
        print(f"    - {col}: {count}")
    print()
    print("  Salidas generadas:")
    print(f"    Dataset:      {OUTPUT_FILE}")
    print(f"    Log cambios:  {NORMALIZATION_LOG_FILE}")
    print(f"    Removidos:    {REMOVED_LOG_FILE}")
    print(f"    Resumen:      {SUMMARY_FILE}")
    print("=" * 60)


def main() -> None:
    config = load_config()
    df = load_data()
    
    df_out, log_records, removed_records, stats = process_dataset(df, config)
    
    save_outputs(df_out, log_records, removed_records, stats, config)
    print_report(stats)


if __name__ == "__main__":
    main()
