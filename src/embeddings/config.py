from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV = PROJECT_ROOT / "data/processed/03_pre_embeddings/dataset_pre_embeddings.csv"
SPLITS_DIR = PROJECT_ROOT / "data/processed/04_splits"
SPLITS_DIR_XL = PROJECT_ROOT / "data/processed/04_splits_xl"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports/04_embeddings"
REPORTS_PREPROCESSING_DIR = REPORTS_DIR / "preprocessing"
REPORTS_PREPROCESSING_XL_DIR = REPORTS_DIR / "preprocessing_xl"
REPORTS_BASELINE_DIR = REPORTS_DIR / "baseline"
REPORTS_V1_DIR = REPORTS_DIR / "v1"
REPORTS_LEGACY_V2_DIR = REPORTS_DIR / "legacy_v2"
REPORTS_CONTROLLED_HN_DIR = REPORTS_DIR / "controlled_hn"
REPORTS_V2_HN_DIR = REPORTS_DIR / "v2_hn_controlled"
REPORTS_V2_HN_HARD_DIR = REPORTS_DIR / "v2_hn_controlled_hard"
REPORTS_EXPLORATORY_DIR = REPORTS_DIR / "exploratory"


def resolve_splits_dir(variant: str = "main") -> Path:
    if variant == "xl":
        return SPLITS_DIR_XL
    return SPLITS_DIR


def resolve_preprocessing_report_dir(variant: str = "main") -> Path:
    if variant == "xl":
        return REPORTS_PREPROCESSING_XL_DIR
    return REPORTS_PREPROCESSING_DIR
