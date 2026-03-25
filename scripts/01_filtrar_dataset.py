"""
01_filtrar_dataset.py
Filtra el CSV original para quedarse solo con filas que tengan
valores válidos en ambas columnas: ESP y SHIWILU.

Entrada:  data/raw/flashcards2.csv
Salida:   data/raw/dataset_esp_shiwilu.csv
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

INPUT_FILE = RAW_DIR / "flashcards2.csv"
OUTPUT_FILE = RAW_DIR / "dataset_esp_shiwilu.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    df_filtrado = df[["ESP", "SHIWILU"]].copy()
    df_filtrado = df_filtrado.dropna(subset=["ESP", "SHIWILU"])
    df_filtrado = df_filtrado[
        (df_filtrado["ESP"].str.strip() != "")
        & (df_filtrado["SHIWILU"].str.strip() != "")
        & (df_filtrado["SHIWILU"].str.strip() != "--")
    ]
    df_filtrado = df_filtrado.reset_index(drop=True)

    df_filtrado.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Filas originales:        {len(df)}")
    print(f"Filas con ambos valores: {len(df_filtrado)}")
    print(f"Archivo guardado:        {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
