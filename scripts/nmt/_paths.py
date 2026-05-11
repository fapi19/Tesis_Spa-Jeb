from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NmtPaths:
    project_root: Path
    variant: str
    splits_dir: Path
    canonical_dir: Path
    filtered_dir: Path
    augmented_dir: Path
    reports_preprocessing_dir: Path
    reports_training_dir: Path
    reports_evaluation_dir: Path
    reports_reranking_dir: Path

    def run_name(self, base: str) -> str:
        if self.variant == "xl" and not base.endswith("_xl"):
            return f"{base}_xl"
        return base


def resolve_paths(project_root: Path, variant: str = "main") -> NmtPaths:
    suffix = "_xl" if variant == "xl" else ""
    return NmtPaths(
        project_root=project_root,
        variant=variant,
        splits_dir=project_root / "data" / "processed" / f"04_splits{suffix}",
        canonical_dir=project_root / "data" / "processed" / f"05_nmt_canonical{suffix}",
        filtered_dir=project_root / "data" / "processed" / f"06_nmt_filtered{suffix}",
        augmented_dir=project_root / "data" / "processed" / f"07_nmt_augmented{suffix}",
        reports_preprocessing_dir=project_root / "reports" / "05_nmt" / f"preprocessing{suffix}",
        reports_training_dir=project_root / "reports" / "05_nmt" / f"training{suffix}",
        reports_evaluation_dir=project_root / "reports" / "05_nmt" / f"evaluation{suffix}",
        reports_reranking_dir=project_root / "reports" / "05_nmt" / f"reranking{suffix}",
    )
