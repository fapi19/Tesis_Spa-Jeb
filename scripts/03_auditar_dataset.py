"""
03_auditar_dataset.py
Auditoría estructural del corpus español-shiwilu.

Detecta problemas potenciales y genera reportes detallados.
Produce el dataset final listo para embeddings.

Entrada:  data/intermediate/dataset_auditado.csv

Salida:   reports/audit_problem_rows.csv
          reports/audit_summary.json
          data/processed/dataset_pre_embeddings.csv
"""

import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = INTERMEDIATE_DIR / "dataset_auditado.csv"
PROBLEM_ROWS_FILE = REPORTS_DIR / "audit_problem_rows.csv"
SUMMARY_FILE = REPORTS_DIR / "audit_summary.json"
OUTPUT_FILE = PROCESSED_DIR / "dataset_pre_embeddings.csv"

PARENTHESIS_PATTERN = re.compile(r"\(.*?\)")
GLOSS_MARKERS = re.compile(r"[\[\]<>{}]")
SUSPICIOUS_CHARS = re.compile(r"[^\w\s\'\'\'\-áéíóúüñÁÉÍÓÚÜÑ.,;:¡!¿?]", re.UNICODE)
NUMERIC_DOMINANT = re.compile(r"^\d+$")


def load_data() -> pd.DataFrame:
    """Carga dataset auditado."""
    return pd.read_csv(INPUT_FILE, encoding="utf-8-sig")


def check_empty_rows(df: pd.DataFrame) -> list[dict]:
    """Detecta filas con campos vacíos."""
    problems = []
    
    for idx, row in df.iterrows():
        issues = []
        
        if pd.isna(row["ESP_original"]) or str(row["ESP_original"]).strip() == "":
            issues.append("ESP_original vacío")
        if pd.isna(row["SHIWILU_original"]) or str(row["SHIWILU_original"]).strip() == "":
            issues.append("SHIWILU_original vacío")
        if pd.isna(row["ESP_normalizado"]) or str(row["ESP_normalizado"]).strip() == "":
            issues.append("ESP_normalizado vacío")
        if pd.isna(row["SHIWILU_normalizado"]) or str(row["SHIWILU_normalizado"]).strip() == "":
            issues.append("SHIWILU_normalizado vacío")
        
        if issues:
            problems.append({
                "pair_id": row["pair_id"],
                "problem_type": "empty_field",
                "details": "; ".join(issues),
                "ESP_original": row["ESP_original"],
                "SHIWILU_original": row["SHIWILU_original"],
                "ESP_normalizado": row["ESP_normalizado"],
                "SHIWILU_normalizado": row["SHIWILU_normalizado"]
            })
    
    return problems


def check_duplicates(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Detecta duplicados exactos por par normalizado.
    
    Returns:
        - Lista de problemas
        - Diccionario con estadísticas de duplicados
    """
    problems = []
    
    dup_mask = df.duplicated(subset=["ESP_normalizado", "SHIWILU_normalizado"], keep=False)
    duplicates_df = df[dup_mask].copy()
    
    if duplicates_df.empty:
        return problems, {"total_duplicate_groups": 0, "total_duplicate_rows": 0}
    
    groups = duplicates_df.groupby(["ESP_normalizado", "SHIWILU_normalizado"])
    
    for (esp, shi), group in groups:
        pair_ids = group["pair_id"].tolist()
        for pair_id in pair_ids[1:]:
            row = df[df["pair_id"] == pair_id].iloc[0]
            problems.append({
                "pair_id": pair_id,
                "problem_type": "exact_duplicate",
                "details": f"Duplicado de {pair_ids[0]} (grupo de {len(pair_ids)})",
                "ESP_original": row["ESP_original"],
                "SHIWILU_original": row["SHIWILU_original"],
                "ESP_normalizado": row["ESP_normalizado"],
                "SHIWILU_normalizado": row["SHIWILU_normalizado"]
            })
    
    stats = {
        "total_duplicate_groups": len(groups),
        "total_duplicate_rows": len(duplicates_df) - len(groups)
    }
    
    return problems, stats


def check_translation_conflicts(df: pd.DataFrame) -> list[dict]:
    """
    Detecta conflictos de traducción:
    - Mismo ESP con múltiples SHIWILU diferentes (1→N)
    - Mismo SHIWILU con múltiples ESP diferentes (N→1)
    """
    problems = []
    
    esp_groups = df.groupby("ESP_normalizado")["SHIWILU_normalizado"].nunique()
    one_to_many = esp_groups[esp_groups > 1]
    
    for esp_norm in one_to_many.index:
        variants = df[df["ESP_normalizado"] == esp_norm]
        shi_variants = variants["SHIWILU_normalizado"].unique().tolist()
        pair_ids = variants["pair_id"].tolist()
        
        for pair_id in pair_ids:
            row = df[df["pair_id"] == pair_id].iloc[0]
            problems.append({
                "pair_id": pair_id,
                "problem_type": "one_to_many_esp",
                "details": f"ESP tiene {len(shi_variants)} traducciones SHIWILU diferentes",
                "ESP_original": row["ESP_original"],
                "SHIWILU_original": row["SHIWILU_original"],
                "ESP_normalizado": row["ESP_normalizado"],
                "SHIWILU_normalizado": row["SHIWILU_normalizado"]
            })
    
    shi_groups = df.groupby("SHIWILU_normalizado")["ESP_normalizado"].nunique()
    many_to_one = shi_groups[shi_groups > 1]
    
    for shi_norm in many_to_one.index:
        variants = df[df["SHIWILU_normalizado"] == shi_norm]
        esp_variants = variants["ESP_normalizado"].unique().tolist()
        pair_ids = variants["pair_id"].tolist()
        
        for pair_id in pair_ids:
            row = df[df["pair_id"] == pair_id].iloc[0]
            existing = [p for p in problems if p["pair_id"] == pair_id]
            if not existing:
                problems.append({
                    "pair_id": pair_id,
                    "problem_type": "many_to_one_shiwilu",
                    "details": f"SHIWILU tiene {len(esp_variants)} traducciones ESP diferentes",
                    "ESP_original": row["ESP_original"],
                    "SHIWILU_original": row["SHIWILU_original"],
                    "ESP_normalizado": row["ESP_normalizado"],
                    "SHIWILU_normalizado": row["SHIWILU_normalizado"]
                })
            else:
                existing[0]["problem_type"] = "bidirectional_conflict"
                existing[0]["details"] += f"; SHIWILU tiene {len(esp_variants)} traducciones ESP"
    
    return problems


def check_length_issues(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Detecta problemas de longitud:
    - Longitudes extremas (muy cortas o muy largas)
    - Desbalance fuerte entre idiomas
    """
    problems = []
    
    df_work = df.copy()
    df_work["esp_words"] = df_work["ESP_normalizado"].str.split().str.len()
    df_work["shi_words"] = df_work["SHIWILU_normalizado"].str.split().str.len()
    df_work["length_ratio"] = df_work["esp_words"] / df_work["shi_words"].replace(0, 1)
    
    esp_mean = df_work["esp_words"].mean()
    esp_std = df_work["esp_words"].std()
    shi_mean = df_work["shi_words"].mean()
    shi_std = df_work["shi_words"].std()
    
    esp_threshold_high = esp_mean + 3 * esp_std
    shi_threshold_high = shi_mean + 3 * shi_std
    
    for idx, row in df_work.iterrows():
        issues = []
        
        if row["esp_words"] <= 0:
            issues.append("ESP sin palabras")
        elif row["esp_words"] > esp_threshold_high:
            issues.append(f"ESP muy largo ({row['esp_words']} palabras, umbral: {esp_threshold_high:.0f})")
        
        if row["shi_words"] <= 0:
            issues.append("SHIWILU sin palabras")
        elif row["shi_words"] > shi_threshold_high:
            issues.append(f"SHIWILU muy largo ({row['shi_words']} palabras, umbral: {shi_threshold_high:.0f})")
        
        if row["esp_words"] > 0 and row["shi_words"] > 0:
            ratio = row["length_ratio"]
            if ratio > 3.0:
                issues.append(f"ESP mucho más largo que SHIWILU (ratio: {ratio:.2f})")
            elif ratio < 0.33:
                issues.append(f"SHIWILU mucho más largo que ESP (ratio: {ratio:.2f})")
        
        if issues:
            orig_row = df.iloc[idx]
            problems.append({
                "pair_id": orig_row["pair_id"],
                "problem_type": "length_issue",
                "details": "; ".join(issues),
                "ESP_original": orig_row["ESP_original"],
                "SHIWILU_original": orig_row["SHIWILU_original"],
                "ESP_normalizado": orig_row["ESP_normalizado"],
                "SHIWILU_normalizado": orig_row["SHIWILU_normalizado"]
            })
    
    stats = {
        "esp_words": {
            "min": int(df_work["esp_words"].min()),
            "max": int(df_work["esp_words"].max()),
            "mean": round(esp_mean, 2),
            "std": round(esp_std, 2),
            "median": int(df_work["esp_words"].median())
        },
        "shi_words": {
            "min": int(df_work["shi_words"].min()),
            "max": int(df_work["shi_words"].max()),
            "mean": round(shi_mean, 2),
            "std": round(shi_std, 2),
            "median": int(df_work["shi_words"].median())
        },
        "length_ratio": {
            "min": round(df_work["length_ratio"].min(), 2),
            "max": round(df_work["length_ratio"].max(), 2),
            "mean": round(df_work["length_ratio"].mean(), 2)
        }
    }
    
    return problems, stats


def check_suspicious_content(df: pd.DataFrame) -> list[dict]:
    """
    Detecta contenido sospechoso:
    - Caracteres inusuales
    - Paréntesis/glosas
    - Contenido numérico dominante
    """
    problems = []
    
    for idx, row in df.iterrows():
        issues = []
        
        esp_orig = str(row["ESP_original"])
        shi_orig = str(row["SHIWILU_original"])
        
        if PARENTHESIS_PATTERN.search(esp_orig):
            issues.append("ESP contiene paréntesis (posible glosa)")
        if PARENTHESIS_PATTERN.search(shi_orig):
            issues.append("SHIWILU contiene paréntesis (posible glosa)")
        
        if GLOSS_MARKERS.search(esp_orig):
            issues.append("ESP contiene marcadores de glosa []<>{}")
        if GLOSS_MARKERS.search(shi_orig):
            issues.append("SHIWILU contiene marcadores de glosa []<>{}")
        
        esp_suspicious = SUSPICIOUS_CHARS.findall(esp_orig)
        if esp_suspicious:
            unique_chars = list(set(esp_suspicious))[:5]
            issues.append(f"ESP contiene caracteres sospechosos: {unique_chars}")
        
        shi_suspicious = SUSPICIOUS_CHARS.findall(shi_orig)
        if shi_suspicious:
            unique_chars = list(set(shi_suspicious))[:5]
            issues.append(f"SHIWILU contiene caracteres sospechosos: {unique_chars}")
        
        esp_clean = re.sub(r"\s+", "", esp_orig)
        shi_clean = re.sub(r"\s+", "", shi_orig)
        
        if esp_clean and NUMERIC_DOMINANT.match(esp_clean):
            issues.append("ESP es solo números")
        if shi_clean and NUMERIC_DOMINANT.match(shi_clean):
            issues.append("SHIWILU es solo números")
        
        if issues:
            problems.append({
                "pair_id": row["pair_id"],
                "problem_type": "suspicious_content",
                "details": "; ".join(issues),
                "ESP_original": row["ESP_original"],
                "SHIWILU_original": row["SHIWILU_original"],
                "ESP_normalizado": row["ESP_normalizado"],
                "SHIWILU_normalizado": row["SHIWILU_normalizado"]
            })
    
    return problems


def calculate_vocabulary_stats(df: pd.DataFrame) -> dict:
    """Calcula estadísticas de vocabulario por idioma."""
    esp_words = []
    shi_words = []
    
    for text in df["ESP_normalizado"]:
        esp_words.extend(str(text).split())
    for text in df["SHIWILU_normalizado"]:
        shi_words.extend(str(text).split())
    
    esp_counter = Counter(esp_words)
    shi_counter = Counter(shi_words)
    
    return {
        "ESP": {
            "total_tokens": len(esp_words),
            "unique_tokens": len(esp_counter),
            "type_token_ratio": round(len(esp_counter) / max(len(esp_words), 1), 4),
            "top_20_words": esp_counter.most_common(20),
            "hapax_legomena": sum(1 for w, c in esp_counter.items() if c == 1)
        },
        "SHIWILU": {
            "total_tokens": len(shi_words),
            "unique_tokens": len(shi_counter),
            "type_token_ratio": round(len(shi_counter) / max(len(shi_words), 1), 4),
            "top_20_words": shi_counter.most_common(20),
            "hapax_legomena": sum(1 for w, c in shi_counter.items() if c == 1)
        }
    }


def consolidate_problems(all_problems: list[dict]) -> pd.DataFrame:
    """
    Consolida problemas por pair_id, agregando múltiples tipos de problema.
    """
    if not all_problems:
        return pd.DataFrame(columns=[
            "pair_id", "problem_types", "all_details",
            "ESP_original", "SHIWILU_original",
            "ESP_normalizado", "SHIWILU_normalizado"
        ])
    
    problems_by_id = {}
    for p in all_problems:
        pid = p["pair_id"]
        if pid not in problems_by_id:
            problems_by_id[pid] = {
                "pair_id": pid,
                "problem_types": [],
                "all_details": [],
                "ESP_original": p["ESP_original"],
                "SHIWILU_original": p["SHIWILU_original"],
                "ESP_normalizado": p["ESP_normalizado"],
                "SHIWILU_normalizado": p["SHIWILU_normalizado"]
            }
        problems_by_id[pid]["problem_types"].append(p["problem_type"])
        problems_by_id[pid]["all_details"].append(p["details"])
    
    rows = []
    for pid, data in problems_by_id.items():
        rows.append({
            "pair_id": data["pair_id"],
            "problem_types": "|".join(set(data["problem_types"])),
            "all_details": " || ".join(data["all_details"]),
            "ESP_original": data["ESP_original"],
            "SHIWILU_original": data["SHIWILU_original"],
            "ESP_normalizado": data["ESP_normalizado"],
            "SHIWILU_normalizado": data["SHIWILU_normalizado"]
        })
    
    return pd.DataFrame(rows)


def generate_final_dataset(df: pd.DataFrame, problem_ids: set[str]) -> pd.DataFrame:
    """
    Genera dataset final para embeddings.
    
    Por defecto NO excluye filas problemáticas automáticamente;
    solo las marca. La exclusión debe ser decisión del investigador.
    """
    df_final = df.copy()
    df_final["has_audit_flags"] = df_final["pair_id"].isin(problem_ids)
    
    return df_final


def save_outputs(
    df_problems: pd.DataFrame,
    df_final: pd.DataFrame,
    summary: dict
) -> None:
    """Guarda todas las salidas de auditoría."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df_problems.to_csv(PROBLEM_ROWS_FILE, index=False, encoding="utf-8-sig")
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def print_report(summary: dict) -> None:
    """Imprime reporte de auditoría en consola."""
    print("=" * 70)
    print("  ETAPA 03: AUDITORÍA DEL CORPUS")
    print("=" * 70)
    print()
    print(f"  Entrada:                    {INPUT_FILE}")
    print(f"  Total filas:                {summary['total_rows']}")
    print(f"  Filas con problemas:        {summary['rows_with_problems']}")
    print(f"  Filas limpias:              {summary['total_rows'] - summary['rows_with_problems']}")
    print()
    print("  PROBLEMAS DETECTADOS:")
    print("  " + "-" * 50)
    
    for ptype, count in summary["problems_by_type"].items():
        print(f"    {ptype}: {count}")
    
    print()
    print("  ESTADÍSTICAS DE LONGITUD:")
    print("  " + "-" * 50)
    length = summary["length_stats"]
    print(f"    ESP:     min={length['esp_words']['min']}, max={length['esp_words']['max']}, "
          f"media={length['esp_words']['mean']:.1f}, mediana={length['esp_words']['median']}")
    print(f"    SHIWILU: min={length['shi_words']['min']}, max={length['shi_words']['max']}, "
          f"media={length['shi_words']['mean']:.1f}, mediana={length['shi_words']['median']}")
    print(f"    Ratio ESP/SHIWILU: min={length['length_ratio']['min']:.2f}, "
          f"max={length['length_ratio']['max']:.2f}, media={length['length_ratio']['mean']:.2f}")
    
    print()
    print("  VOCABULARIO:")
    print("  " + "-" * 50)
    vocab = summary["vocabulary_stats"]
    print(f"    ESP:     {vocab['ESP']['unique_tokens']} tipos / {vocab['ESP']['total_tokens']} tokens "
          f"(TTR: {vocab['ESP']['type_token_ratio']:.4f})")
    print(f"    SHIWILU: {vocab['SHIWILU']['unique_tokens']} tipos / {vocab['SHIWILU']['total_tokens']} tokens "
          f"(TTR: {vocab['SHIWILU']['type_token_ratio']:.4f})")
    print(f"    Hapax ESP: {vocab['ESP']['hapax_legomena']}, SHIWILU: {vocab['SHIWILU']['hapax_legomena']}")
    
    if summary["duplicate_stats"]["total_duplicate_groups"] > 0:
        print()
        print("  DUPLICADOS:")
        print("  " + "-" * 50)
        dup = summary["duplicate_stats"]
        print(f"    Grupos de duplicados: {dup['total_duplicate_groups']}")
        print(f"    Filas duplicadas:     {dup['total_duplicate_rows']}")
    
    print()
    print("  SALIDAS GENERADAS:")
    print("  " + "-" * 50)
    print(f"    Problemas:      {PROBLEM_ROWS_FILE}")
    print(f"    Resumen JSON:   {SUMMARY_FILE}")
    print(f"    Dataset final:  {OUTPUT_FILE}")
    print()
    print("  NOTA: Las filas problemáticas están MARCADAS pero NO eliminadas.")
    print("  La columna 'has_audit_flags' indica si la fila tiene alertas.")
    print("  El investigador debe decidir qué filas excluir según criterio.")
    print("=" * 70)


def main() -> None:
    df = load_data()
    
    all_problems = []
    
    empty_problems = check_empty_rows(df)
    all_problems.extend(empty_problems)
    
    duplicate_problems, duplicate_stats = check_duplicates(df)
    all_problems.extend(duplicate_problems)
    
    conflict_problems = check_translation_conflicts(df)
    all_problems.extend(conflict_problems)
    
    length_problems, length_stats = check_length_issues(df)
    all_problems.extend(length_problems)
    
    suspicious_problems = check_suspicious_content(df)
    all_problems.extend(suspicious_problems)
    
    vocab_stats = calculate_vocabulary_stats(df)
    
    df_problems = consolidate_problems(all_problems)
    problem_ids = set(df_problems["pair_id"]) if not df_problems.empty else set()
    
    problems_by_type = {}
    for p in all_problems:
        ptype = p["problem_type"]
        problems_by_type[ptype] = problems_by_type.get(ptype, 0) + 1
    
    summary = {
        "pipeline": "03_auditar_dataset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_FILE),
        "total_rows": len(df),
        "rows_with_problems": len(problem_ids),
        "problems_by_type": problems_by_type,
        "duplicate_stats": duplicate_stats,
        "length_stats": length_stats,
        "vocabulary_stats": vocab_stats
    }
    
    df_final = generate_final_dataset(df, problem_ids)
    
    save_outputs(df_problems, df_final, summary)
    print_report(summary)


if __name__ == "__main__":
    main()
