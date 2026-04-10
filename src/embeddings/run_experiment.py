"""
Orquestador de experimentos de embeddings bilingüe español-shiwilu.

Uso:
    python -m src.embeddings.run_experiment E0           # solo E0
    python -m src.embeddings.run_experiment E0 E1        # E0 y E1
    python -m src.embeddings.run_experiment --all        # todos los experimentos

Experimentos:
    E0  Unigram + contrastive
    E1  BPE + contrastive
    E2  Suffix-aware + contrastive
    E3  Suffix-aware + contrastive + alignment
    E4  Suffix-aware + focal contrastive + alignment
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import SPLITS_DIR, MODELS_DIR


@dataclass
class ExperimentConfig:
    name: str
    description: str
    tokenizer_type: str          # "unigram" | "bpe"
    suffix_aware: bool           # use suffix-aware corpus
    use_focal: bool              # use focal InfoNCE instead of standard InfoNCE
    use_alignment: bool          # add bilingual alignment loss
    vocab_size: int = 4000
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    d_model: int = 192
    nhead: int = 4
    num_layers: int = 2
    ff_dim: int = 768
    dropout: float = 0.2
    contrastive_weight: float = 1.0
    alignment_weight: float = 0.3
    temperature: float = 0.05
    gamma: float = 2.0

    @property
    def exp_dir(self) -> Path:
        return MODELS_DIR / "embeddings" / self.name

    @property
    def corpus_path(self) -> Path:
        if self.suffix_aware:
            return SPLITS_DIR / "all_text_suffix_aware.txt"
        return SPLITS_DIR / "all_text.txt"

    @property
    def train_jsonl(self) -> Path:
        if self.suffix_aware:
            return SPLITS_DIR / "train_pairs_suffix_aware.jsonl"
        return SPLITS_DIR / "train_pairs.jsonl"

    @property
    def val_jsonl(self) -> Path:
        if self.suffix_aware:
            return SPLITS_DIR / "val_pairs_suffix_aware.jsonl"
        return SPLITS_DIR / "val_pairs.jsonl"

    @property
    def tokenizer_prefix(self) -> Path:
        return self.exp_dir / "tokenizer"

    @property
    def sp_model_path(self) -> Path:
        return Path(f"{self.tokenizer_prefix}.model")

    @property
    def checkpoint_path(self) -> Path:
        return self.exp_dir / "checkpoint.pt"


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "E0": ExperimentConfig(
        name="E0_unigram_contrastive",
        description="Unigram + contrastive — baseline subword con InfoNCE",
        tokenizer_type="unigram",
        suffix_aware=False,
        use_focal=False,
        use_alignment=False,
    ),
    "E1": ExperimentConfig(
        name="E1_bpe_contrastive",
        description="BPE + contrastive — comparar unigram vs BPE",
        tokenizer_type="bpe",
        suffix_aware=False,
        use_focal=False,
        use_alignment=False,
    ),
    "E2": ExperimentConfig(
        name="E2_suffix_contrastive",
        description="Suffix-aware + contrastive — señal morfológica del shiwilu",
        tokenizer_type="unigram",
        suffix_aware=True,
        use_focal=False,
        use_alignment=False,
    ),
    "E3": ExperimentConfig(
        name="E3_suffix_contrastive_alignment",
        description="Suffix-aware + contrastive + alignment — morfología + alineación bilingüe",
        tokenizer_type="unigram",
        suffix_aware=True,
        use_focal=False,
        use_alignment=True,
        contrastive_weight=1.0,
        alignment_weight=0.3,
    ),
    "E4": ExperimentConfig(
        name="E4_suffix_focal_alignment",
        description="Suffix-aware + focal contrastive + alignment — variante más fuerte",
        tokenizer_type="unigram",
        suffix_aware=True,
        use_focal=True,
        use_alignment=True,
        batch_size=32,  #Added for testing purposes
        contrastive_weight=1.0,
        alignment_weight=0.3,
        gamma=2.0,
    ),
}


def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def prepare_shared_data() -> None:
    """Run prepare_data (always) and suffix pipeline (if needed)."""
    from .prepare_data import main as prepare_main

    _banner("Paso 0: preparar datos (CSV → JSONL + all_text)")
    prepare_main()

    if not (SPLITS_DIR / "shiwilu_suffixes.json").exists():
        from .mine_suffixes import main as mine_main

        _banner("Paso 0.1: minar sufijos shiwilu")
        mine_main()

    if not (SPLITS_DIR / "train_pairs_suffix_aware.jsonl").exists():
        from .build_suffix_aware_corpus import main as suffix_main

        _banner("Paso 0.2: construir corpus suffix-aware")
        suffix_main()


def run_experiment(cfg: ExperimentConfig) -> None:
    from .train_tokenizer import train_tokenizer
    from .train_embedding_model import train

    _banner(f"{cfg.name}: {cfg.description}")

    if cfg.suffix_aware:
        if not cfg.train_jsonl.exists():
            from .mine_suffixes import main as mine_main
            from .build_suffix_aware_corpus import main as suffix_main

            mine_main()
            suffix_main()

    # --- tokenizer ---
    if cfg.sp_model_path.exists():
        print(f"  Tokenizer ya existe: {cfg.sp_model_path}")
    else:
        print(f"  Entrenando tokenizer ({cfg.tokenizer_type}, vocab={cfg.vocab_size})...")
        train_tokenizer(
            input_path=cfg.corpus_path,
            model_prefix=cfg.tokenizer_prefix,
            vocab_size=cfg.vocab_size,
            model_type=cfg.tokenizer_type,
        )

    # --- modelo ---
    print(
        f"  Entrenando modelo (epochs={cfg.epochs}, "
        f"focal={cfg.use_focal}, alignment={cfg.use_alignment})..."
    )
    t0 = time.time()
    train(
        train_path=cfg.train_jsonl,
        val_path=cfg.val_jsonl,
        sp_model=cfg.sp_model_path,
        save_path=cfg.checkpoint_path,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        use_focal=cfg.use_focal,
        use_alignment=cfg.use_alignment,
        contrastive_weight=cfg.contrastive_weight,
        alignment_weight=cfg.alignment_weight,
        temperature=cfg.temperature,
        gamma=cfg.gamma,
    )
    elapsed = time.time() - t0
    print(f"  Finalizado en {elapsed / 60:.1f} min → {cfg.checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correr experimentos de embeddings bilingüe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {key}  {cfg.description}" for key, cfg in EXPERIMENTS.items()
        ),
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        choices=list(EXPERIMENTS.keys()),
        help="Experimentos a correr (E0, E1, E2, E3, E4)",
    )
    parser.add_argument("--all", action="store_true", help="Correr todos los experimentos")
    parser.add_argument("--skip-data-prep", action="store_true", help="Saltar preparación de datos")
    args = parser.parse_args()

    if not args.experiments and not args.all:
        parser.print_help()
        sys.exit(1)

    selected = list(EXPERIMENTS.keys()) if args.all else args.experiments

    print(f"Experimentos seleccionados: {', '.join(selected)}")

    if not args.skip_data_prep:
        prepare_shared_data()

    for key in selected:
        run_experiment(EXPERIMENTS[key])

    _banner("Todos los experimentos completados")
    for key in selected:
        cfg = EXPERIMENTS[key]
        print(f"  {key}: {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
