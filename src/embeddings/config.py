from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = PROJECT_ROOT / "data/processed/03_pre_embeddings/dataset_pre_embeddings.csv"
SPLITS_DIR = PROJECT_ROOT / "data/processed/04_splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports/04_embeddings"
