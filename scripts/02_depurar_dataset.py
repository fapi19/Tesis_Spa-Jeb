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
  7. Reporte, gráficas y exportación

Uso:
  poetry run python scripts/02_depurar_dataset.py

Entrada:  data/raw/dataset_esp_shiwilu.csv
Salida:   data/processed/dataset_limpio.csv
          reports/*.png
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = RAW_DIR / "dataset_esp_shiwilu.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_limpio.csv"

PUNCTUATION_PATTERN = re.compile(r"[¡!¿?.,:;\"«»\-—…]")
PARENTHESIS_PATTERN = re.compile(r"\s*\(.*?\)\s*")
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})


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


# ── Gráficas ─────────────────────────────────────────────────────────────────

def grafica_distribucion_longitudes(df: pd.DataFrame) -> None:
    """Histograma de cantidad de palabras por oración en cada idioma."""
    esp_len = df["ESP"].str.split().str.len()
    shi_len = df["SHIWILU"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(esp_len, bins=range(1, esp_len.max() + 2), color="#4A90D9",
                 edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribución de longitudes — Español")
    axes[0].set_xlabel("Palabras por oración")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(shi_len, bins=range(1, shi_len.max() + 2), color="#D97B4A",
                 edgecolor="white", alpha=0.85)
    axes[1].set_title("Distribución de longitudes — Shiwilu")
    axes[1].set_xlabel("Palabras por oración")
    axes[1].set_ylabel("Frecuencia")

    fig.suptitle("Distribución de longitudes (palabras por oración)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "distribucion_longitudes.png")
    plt.close(fig)


def grafica_top_palabras(df: pd.DataFrame, top_n: int = 20) -> None:
    """Barras horizontales con las palabras más frecuentes por idioma."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (col, color, titulo) in enumerate([
        ("ESP", "#4A90D9", "Top 20 palabras — Español"),
        ("SHIWILU", "#D97B4A", "Top 20 palabras — Shiwilu"),
    ]):
        all_words = " ".join(df[col]).split()
        most_common = Counter(all_words).most_common(top_n)
        words, counts = zip(*reversed(most_common))

        axes[idx].barh(words, counts, color=color, edgecolor="white")
        axes[idx].set_title(titulo)
        axes[idx].set_xlabel("Frecuencia")

    fig.suptitle("Palabras más frecuentes por idioma", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "top_palabras.png")
    plt.close(fig)


def grafica_antes_despues(conteos: dict[str, int], total_final: int) -> None:
    """Barras mostrando cuántas filas se afectaron en cada paso."""
    pasos = [
        "Entrada",
        "P1: Espacios",
        "P3: Puntuación",
        "P4: Paréntesis",
        "P5: Unicode",
        "P6: Duplicados",
    ]
    valores = [
        conteos["entrada"],
        conteos["paso_1"],
        conteos["paso_3"],
        conteos["paso_4"],
        conteos["paso_5"],
        conteos["paso_6"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

    colores = ["#4A90D9"] + ["#D97B4A"] * 5
    axes[0].bar(pasos, valores, color=colores, edgecolor="white")
    axes[0].set_ylabel("Filas afectadas / eliminadas")
    axes[0].set_title("Filas afectadas por paso de limpieza")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(
        ["Antes", "Después"],
        [conteos["entrada"], total_final],
        color=["#888888", "#2ECC71"],
        edgecolor="white",
        width=0.5,
    )
    axes[1].set_ylabel("Total de filas")
    axes[1].set_title("Antes vs Después")

    fig.suptitle("Impacto del pipeline de depuración", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "comparacion_antes_despues.png")
    plt.close(fig)


def grafica_correlacion_longitudes(df: pd.DataFrame) -> None:
    """Scatter plot: longitud ESP vs longitud SHIWILU."""
    esp_len = df["ESP"].str.split().str.len()
    shi_len = df["SHIWILU"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(esp_len, shi_len, alpha=0.3, s=20, color="#4A90D9", edgecolors="none")

    max_val = max(esp_len.max(), shi_len.max()) + 1
    ax.plot([0, max_val], [0, max_val], "--", color="#999999", linewidth=1, label="x = y")

    ax.set_xlabel("Palabras en Español")
    ax.set_ylabel("Palabras en Shiwilu")
    ax.set_title("Correlación de longitudes ESP vs SHIWILU", fontsize=13,
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "longitud_correlacion.png")
    plt.close(fig)


# ── Paso 7: Reporte y exportación ────────────────────────────────────────────

def paso_7_exportar_y_reportar(
    df_limpio: pd.DataFrame,
    conteos: dict[str, int],
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_limpio.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    esp_words = df_limpio["ESP"].str.split().str.len()
    shi_words = df_limpio["SHIWILU"].str.split().str.len()

    vocab_esp: set[str] = set()
    vocab_shi: set[str] = set()
    for text in df_limpio["ESP"]:
        vocab_esp.update(text.split())
    for text in df_limpio["SHIWILU"]:
        vocab_shi.update(text.split())

    print("=" * 60)
    print("  REPORTE DE DEPURACIÓN — Dataset ESP-Shiwilu")
    print("=" * 60)
    print()
    print(f"  Filas de entrada:           {conteos['entrada']}")
    print(f"  Filas de salida:            {len(df_limpio)}")
    print(f"  Filas eliminadas (total):   {conteos['entrada'] - len(df_limpio)}")
    print()
    print("  Detalle por paso:")
    print(f"    Paso 1 — Espacios corregidos:        {conteos['paso_1']}")
    print(f"    Paso 2 — Convertidas a minúsculas:    (todas)")
    print(f"    Paso 3 — Puntuación eliminada en:     {conteos['paso_3']} celdas")
    print(f"    Paso 4 — Paréntesis limpiados:        {conteos['paso_4']}")
    print(f"    Paso 5 — Unicode normalizado:          {conteos['paso_5']}")
    print(f"    Paso 6 — Duplicados eliminados:       {conteos['paso_6']}")
    print()
    print("  Distribución de longitudes (palabras por oración):")
    print(f"    ESP     → min: {esp_words.min()}, max: {esp_words.max()}, "
          f"promedio: {esp_words.mean():.1f}")
    print(f"    SHIWILU → min: {shi_words.min()}, max: {shi_words.max()}, "
          f"promedio: {shi_words.mean():.1f}")
    print()
    print(f"  Vocabulario único:")
    print(f"    ESP:     {len(vocab_esp)} palabras únicas")
    print(f"    SHIWILU: {len(vocab_shi)} palabras únicas")
    print()
    print("  Muestra del resultado (primeras 10 filas):")
    print()
    for _, row in df_limpio.head(10).iterrows():
        print(f"    {row['ESP']:40s} → {row['SHIWILU']}")
    print()
    print("  Archivos generados:")
    print(f"    CSV:      {OUTPUT_FILE}")
    print(f"    Gráficas: {REPORTS_DIR}/")
    print("              - distribucion_longitudes.png")
    print("              - top_palabras.png")
    print("              - comparacion_antes_despues.png")
    print("              - longitud_correlacion.png")
    print("=" * 60)

    grafica_distribucion_longitudes(df_limpio)
    grafica_top_palabras(df_limpio)
    grafica_antes_despues(conteos, len(df_limpio))
    grafica_correlacion_longitudes(df_limpio)

    print("\n  Gráficas generadas correctamente.")


# ── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    df = load_data()
    conteos: dict[str, int] = {"entrada": len(df)}

    # Paso 1
    espacios_antes = sum(
        int(df[col].str.contains(r"\s{2,}|^\s|\s$", regex=True).sum())
        for col in ["ESP", "SHIWILU"]
    )
    df = paso_1_trim_espacios(df)
    conteos["paso_1"] = espacios_antes

    # Paso 2
    df = paso_2_lowercase(df)

    # Paso 3
    puntuacion_antes = sum(
        int(df[col].str.contains(PUNCTUATION_PATTERN).sum())
        for col in ["ESP", "SHIWILU"]
    )
    df = paso_3_quitar_puntuacion(df)
    conteos["paso_3"] = puntuacion_antes

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
    paso_7_exportar_y_reportar(df, conteos)


if __name__ == "__main__":
    main()
