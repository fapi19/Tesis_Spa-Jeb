from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = PROJECT_ROOT / "data/processed/03_pre_embeddings/dataset_pre_embeddings.csv"
SPLITS_DIR = PROJECT_ROOT / "data/processed/04_splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports/04_embeddings"
REPORTS_PREPROCESSING_DIR = REPORTS_DIR / "preprocessing"
REPORTS_BASELINE_DIR = REPORTS_DIR / "baseline"
REPORTS_V1_DIR = REPORTS_DIR / "v1"
REPORTS_LEGACY_V2_DIR = REPORTS_DIR / "legacy_v2"
REPORTS_CONTROLLED_HN_DIR = REPORTS_DIR / "controlled_hn"
REPORTS_V2_HN_DIR = REPORTS_DIR / "v2_hn_controlled"
REPORTS_V2_HN_HARD_DIR = REPORTS_DIR / "v2_hn_controlled_hard"
REPORTS_EXPLORATORY_DIR = REPORTS_DIR / "exploratory"
