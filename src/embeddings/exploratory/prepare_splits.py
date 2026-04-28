"""
prepare_splits.py
Preparación de splits reproducibles para entrenamiento de embeddings.

Toma el corpus pre-embeddings y genera splits train/valid/test con
opción de filtrar pares con audit flags.

Uso:
    poetry run python src/embeddings/prepare_splits.py
    poetry run python src/embeddings/prepare_splits.py --clean-only
    poetry run python src/embeddings/prepare_splits.py --seed 123

Salida:
    data/processed/04_splits/train.csv
    data/processed/04_splits/valid.csv
    data/processed/04_splits/test.csv
    reports/04_embeddings/preprocessing/splits_summary.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.embeddings.preprocess_embeddings import preprocess_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "03_pre_embeddings" / "dataset_pre_embeddings.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "04_splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "04_embeddings" / "preprocessing"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Genera splits train/valid/test para entrenamiento de embeddings"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Ruta al CSV del corpus (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Usar solo pares sin audit flags (has_audit_flags == False)"
    )
    return parser.parse_args()


def load_corpus(filepath: Path) -> pd.DataFrame:
    """Carga el corpus y retorna DataFrame."""
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    return pd.read_csv(filepath, encoding="utf-8-sig")


def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra pares sin audit flags."""
    return df[df["has_audit_flags"] == False].copy()  # noqa: E712


def create_splits(
    df: pd.DataFrame,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en train/valid/test (80/10/10).

    Usa shuffle reproducible con la semilla dada.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_df = shuffled.iloc[:n_train]
    valid_df = shuffled.iloc[n_train:n_train + n_valid]
    test_df = shuffled.iloc[n_train + n_valid:]

    return train_df, valid_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """Guarda los splits como CSV."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(SPLITS_DIR / "valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False, encoding="utf-8-sig")


def save_report(
    df_original: pd.DataFrame,
    df_used: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clean_only: bool,
    seed: int,
    start_time: datetime
) -> None:
    """Guarda reporte JSON del split."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    elapsed = datetime.now(timezone.utc) - start_time
    summary = {
        "pipeline": "prepare_splits",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "clean_only": clean_only,
        "total_original": len(df_original),
        "total_used": len(df_used),
        "filtered_out": len(df_original) - len(df_used),
        "splits": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
        },
        "ratios": {
            "train": f"{len(train_df) / len(df_used):.2%}",
            "valid": f"{len(valid_df) / len(df_used):.2%}",
            "test": f"{len(test_df) / len(df_used):.2%}",
        },
        "sources_distribution": {
            split_name: df_split["source"].value_counts().to_dict()
            for split_name, df_split in [
                ("train", train_df),
                ("valid", valid_df),
                ("test", test_df),
            ]
        },
        "elapsed_seconds": elapsed.total_seconds()
    }

    with open(REPORTS_DIR / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_report(
    df_original: pd.DataFrame,
    df_used: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clean_only: bool,
    seed: int,
    start_time: datetime
) -> None:
    """Imprime reporte en consola."""
    elapsed = datetime.now(timezone.utc) - start_time

    print("\n" + "=" * 70)
    print("  ETAPA 04c: PREPARACIÓN DE SPLITS")
    print("=" * 70)
    print()
    print(f"  Semilla:              {seed}")
    print(f"  Solo limpios:         {'Sí' if clean_only else 'No'}")
    print(f"  Total original:       {len(df_original):,}")
    print(f"  Total usado:          {len(df_used):,}")
    if clean_only:
        print(f"  Filtrados (audit):    {len(df_original) - len(df_used):,}")
    print(f"  Tiempo de ejecución:  {elapsed.total_seconds():.2f}s")

    print()
    print("  SPLITS:")
    print("  " + "-" * 50)
    print(f"    Train:  {len(train_df):,}  ({len(train_df) / len(df_used):.1%})")
    print(f"    Valid:  {len(valid_df):,}  ({len(valid_df) / len(df_used):.1%})")
    print(f"    Test:   {len(test_df):,}  ({len(test_df) / len(df_used):.1%})")

    print()
    print("  DISTRIBUCIÓN POR FUENTE:")
    print("  " + "-" * 50)
    for name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        counts = split_df["source"].value_counts()
        parts = [f"{src}: {cnt}" for src, cnt in counts.items()]
        print(f"    {name:<7} {', '.join(parts)}")

    print()
    print("  SALIDAS GENERADAS:")
    print("  " + "-" * 50)
    print(f"    Train CSV:    {SPLITS_DIR / 'train.csv'}")
    print(f"    Valid CSV:    {SPLITS_DIR / 'valid.csv'}")
    print(f"    Test CSV:     {SPLITS_DIR / 'test.csv'}")
    print(f"    Resumen JSON: {REPORTS_DIR / 'splits_summary.json'}")
    print("=" * 70)


def main() -> None:
    """Función principal."""
    args = parse_args()
    if args.clean_only:
        print("Aviso: --clean-only se conserva por compatibilidad, pero el pipeline canónico audita sin descartar flags automáticamente.")

    splits = preprocess_embeddings(args.data, seed=args.seed)
    print("prepare_splits ahora delega al pipeline canónico de embeddings.")
    print(f"Train: {len(splits['train'])}")
    print(f"Valid: {len(splits['valid'])}")
    print(f"Test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
