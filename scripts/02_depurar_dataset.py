"""
02_depurar_dataset.py
Pipeline de depuración del dataset español-shiwilu para embeddings.

7 pasos:
  1. Trim y normalización de espacios
  2. Conversión a minúsculas
  3. Eliminación de signos de puntuación
  4. Limpieza de paréntesis explicativos
  5. Normalización Unicode (NFC)
  6. Eliminación de duplicados exactos
  7. Reporte y exportación

Entrada:  data/raw/dataset_esp_shiwilu.csv
Salida:   data/processed/dataset_limpio.csv
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = RAW_DIR / "dataset_esp_shiwilu.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_limpio.csv"

PUNCTUATION_PATTERN = re.compile(r"[¡!¿?.,:;\"«»\-—…]")
PARENTHESIS_PATTERN = re.compile(r"\s*\(.*?\)\s*")
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)
    df["ESP"] = df["ESP"].astype(str)
    df["SHIWILU"] = df["SHIWILU"].astype(str)
    return df


# ── Paso 1: Trim y espacios ─────────────────────────────────────────────────

def paso_1_trim_espacios(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ESP", "SHIWILU"]:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(" , ", ", ", regex=False)
        df[col] = df[col].apply(lambda x: MULTI_SPACE_PATTERN.sub(" ", x))
    return df


# ── Paso 2: Minúsculas ──────────────────────────────────────────────────────

def paso_2_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ESP", "SHIWILU"]:
        df[col] = df[col].str.lower()
    return df


# ── Paso 3: Eliminar puntuación ─────────────────────────────────────────────

def paso_3_quitar_puntuacion(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ESP", "SHIWILU"]:
        df[col] = df[col].apply(lambda x: PUNCTUATION_PATTERN.sub("", x))
        df[col] = df[col].str.strip()
    return df


# ── Paso 4: Limpiar paréntesis explicativos ──────────────────────────────────

def paso_4_limpiar_parentesis(df: pd.DataFrame) -> pd.DataFrame:
    df["ESP"] = df["ESP"].apply(lambda x: PARENTHESIS_PATTERN.sub("", x).strip())
    return df


# ── Paso 5: Normalización Unicode ────────────────────────────────────────────

def paso_5_normalizar_unicode(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ESP", "SHIWILU"]:
        df[col] = df[col].apply(lambda x: unicodedata.normalize("NFC", x))
    return df


# ── Paso 6: Eliminar duplicados exactos ──────────────────────────────────────

def paso_6_eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["ESP", "SHIWILU"]).reset_index(drop=True)


# ── Paso 7: Reporte y exportación ────────────────────────────────────────────

def paso_7_exportar_y_reportar(
    df_original: pd.DataFrame,
    df_limpio: pd.DataFrame,
    conteos: dict[str, int],
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_limpio.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    esp_words = df_limpio["ESP"].str.split().str.len()
    shi_words = df_limpio["SHIWILU"].str.split().str.len()

    print("=" * 60)
    print("  REPORTE DE DEPURACIÓN — Dataset ESP-Shiwilu")
    print("=" * 60)
    print()
    print(f"  Filas de entrada:           {conteos['entrada']}")
    print(f"  Filas de salida:            {len(df_limpio)}")
    print(f"  Filas eliminadas (total):   {conteos['entrada'] - len(df_limpio)}")
    print()
    print("  Detalle por paso:")
    print(f"    Paso 1 — Espacios corregidos:       {conteos['paso_1']}")
    print(f"    Paso 2 — Convertidas a minúsculas:   (todas)")
    print(f"    Paso 3 — Puntuación eliminada:       {conteos['paso_3']}")
    print(f"    Paso 4 — Paréntesis limpiados:       {conteos['paso_4']}")
    print(f"    Paso 5 — Unicode normalizado:         {conteos['paso_5']}")
    print(f"    Paso 6 — Duplicados eliminados:      {conteos['paso_6']}")
    print()
    print("  Distribución de longitudes (palabras por oración):")
    print(f"    ESP     → min: {esp_words.min()}, max: {esp_words.max()}, "
          f"promedio: {esp_words.mean():.1f}")
    print(f"    SHIWILU → min: {shi_words.min()}, max: {shi_words.max()}, "
          f"promedio: {shi_words.mean():.1f}")
    print()
    print("  Vocabulario único:")
    vocab_esp = set()
    vocab_shi = set()
    for text in df_limpio["ESP"]:
        vocab_esp.update(text.split())
    for text in df_limpio["SHIWILU"]:
        vocab_shi.update(text.split())
    print(f"    ESP:     {len(vocab_esp)} palabras únicas")
    print(f"    SHIWILU: {len(vocab_shi)} palabras únicas")
    print()
    print("  Muestra del resultado (primeras 10 filas):")
    print()
    for _, row in df_limpio.head(10).iterrows():
        print(f"    {row['ESP']:40s} → {row['SHIWILU']}")
    print()
    print(f"  Archivo guardado: {OUTPUT_FILE}")
    print("=" * 60)


# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    df = load_data()
    conteos: dict[str, int] = {"entrada": len(df)}

    # Paso 1
    espacios_antes = sum(
        df[col].str.contains(r"\s{2,}|^\s|\s$", regex=True).sum()
        for col in ["ESP", "SHIWILU"]
    )
    df = paso_1_trim_espacios(df)
    conteos["paso_1"] = int(espacios_antes)

    # Paso 2
    df = paso_2_lowercase(df)

    # Paso 3
    puntuacion_antes = sum(
        df[col].str.contains(PUNCTUATION_PATTERN).sum()
        for col in ["ESP", "SHIWILU"]
    )
    df = paso_3_quitar_puntuacion(df)
    conteos["paso_3"] = int(puntuacion_antes)

    # Paso 4
    parentesis_antes = int(df["ESP"].str.contains(r"\(", regex=True).sum())
    df = paso_4_limpiar_parentesis(df)
    conteos["paso_4"] = parentesis_antes

    # Paso 5
    unicode_antes = 0
    for col in ["ESP", "SHIWILU"]:
        for text in df[col]:
            if unicodedata.normalize("NFC", text) != text:
                unicode_antes += 1
    df = paso_5_normalizar_unicode(df)
    conteos["paso_5"] = unicode_antes

    # Paso 6
    filas_antes_dup = len(df)
    df = paso_6_eliminar_duplicados(df)
    conteos["paso_6"] = filas_antes_dup - len(df)

    # Paso 7
    df_original = load_data()
    paso_7_exportar_y_reportar(df_original, df, conteos)


if __name__ == "__main__":
    main()
