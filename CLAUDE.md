# CLAUDE.md — Project Knowledge for Spanish-Shiwilu NMT Thesis

This file is the durable project guide for Claude sessions. Read it before doing
anything substantive. Update it when you learn something durable about the
project (file moved, new convention, recurring bug).

## Project at a glance

- **Goal**: Build a bidirectional NMT system between **Spanish (`spa`)** and
  **Shiwilu / Jebero (`shw`)**, a low-resource Cahuapanan language from the
  Peruvian Amazon.
- **Approach**: NLLB-200-distilled-600M extended with `shw_Latn` as a new
  language code, fine-tuned with LoRA (and now DoRA + LoRA+) on a small
  parallel corpus (~3-7k pairs depending on variant), augmented with mining
  + round-trip backtranslation. Reranked at inference with a fine-tuned
  multilingual sentence-transformer (E5-base bidirectional + iterative hard
  negatives).
- **Stage**: Both pipelines (embeddings + NMT) are functional. Best models
  shipped under `models/sentence_transformers/v3_iterative_hn_e5_base_bidirectional_xl/`
  and `models/nmt/nllb_bidi_lora_v1_bt_xl/`. Currently mid-ablation studies
  (DoRA, LoRA+, iterative BT, Two DoRAs).

## Hardware + environment

- **OS**: Windows 11 Pro 10.0.26200
- **Shell**: PowerShell *and* Bash (Bash tool available, but use PowerShell
  syntax for shell-only operations like env vars).
- **GPU**: 1× NVIDIA RTX 5060 Ti (Blackwell, sm_120, 16 GB VRAM)
- **CUDA**: driver 596.36 (CUDA 13.2-capable), torch wheel `torch==2.7.1+cu128`
- **Python**: 3.12.0

### Two virtualenvs — pick the right one

| Env | Path | Use for | torch | datasets | sentence_transformers | peft |
|---|---|---|---|---|---|---|
| **`.venv-nmt`** ✅ default | `.venv-nmt/Scripts/python` | **All NMT + embeddings work** | 2.7.1+cu128 (CUDA) | 3.6.0 | 5.4.1 | 0.16.0 (DoRA + LoRA+) |
| `.venv` | `.venv/Scripts/python` | Avoid — CPU-only torch | 2.11.0+cpu | (installed mid-session, but no CUDA) | 5.4.1 | 0.19.1 |

**Always use `.venv-nmt` for any training, eval, or generation.** `.venv` was
the original poetry-managed env but its torch is CPU-only, so a 10-epoch
fine-tune that takes 11 min on GPU takes 4-8 h on CPU.

### Things that DO NOT work on this Windows machine

- `torch.compile` — needs Triton (Linux-only). Don't add `--compile` flag to
  training; it crashes with `TritonMissing`.
- HuggingFace symlink cache — falls back to copies (just a warning, not an
  error). `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence.
- `gcloud auth login` and other interactive logins from inside the agent —
  ask the user to run `! <command>` themselves.

## Repository layout (only the parts you actually need)

```
Desarrollo/
├── config/
│   ├── sources.json              # Registry of raw CSV sources for data prep
│   ├── normalization_rules.json  # Cleaning rules for stage 02
│   └── nmt/
│       ├── training.yaml         # NMT training hyperparameters (LoRA r=16, α=32 by default)
│       ├── inference.yaml        # Beam search + generation config
│       ├── eval.yaml             # COMET / BERTScore config for final eval
│       ├── filter.yaml           # Semantic filter thresholds
│       └── reranker.yaml         # 0.7 trans / 0.3 semantic + alpha sweep
├── data/
│   ├── raw/                      # Original sources (flashcards, PDFs, narratives)
│   ├── intermediate/             # Per-stage outputs (01_filtrado, 01b_unificado, 02_normalizado, 00_pdf)
│   ├── processed/
│   │   ├── 03_pre_embeddings/dataset_pre_embeddings.csv     # Audited dataset
│   │   ├── 04_splits/            # main variant embedding splits
│   │   ├── 04_splits_xl/         # xl variant (larger, see below)
│   │   ├── 05_nmt_canonical[/_xl]/    # NMT canonical CSVs
│   │   ├── 06_nmt_filtered[/_xl]/     # NMT filtered + FAISS indexes
│   │   └── 07_nmt_augmented[/_xl]/    # BT, mining, morph augmentation outputs
├── scripts/
│   ├── 00_extraer_dataset_pdf.py        # PDF extraction (run once; output already exists)
│   ├── 01_filtrar_dataset.py             # Per-source NaN/empty filtering
│   ├── 01b_unificar_fuentes.py           # Combine all sources
│   ├── 02_depurar_dataset.py             # Normalization (lowercase, trim, etc.)
│   ├── 03_auditar_dataset.py             # Audit + emit pre_embeddings dataset
│   ├── generate_nmt_tables.py            # Generate LaTeX tables for thesis
│   └── nmt/
│       ├── _paths.py                     # resolve_paths(root, variant) — the canonical
│       │                                 # router for main vs xl directories
│       ├── 10_canonicalize_dataset.py
│       ├── 20_semantic_filter.py
│       ├── 21_build_faiss.py
│       ├── 22_train_sentencepiece.py
│       ├── 30_train_lora.py              # Phase 4: train v0 NMT
│       ├── 40_evaluate.py                # Phase 5: full eval (BLEU, chrF, BERTScore, COMET)
│       ├── 41_rare_token_eval.py         # Phase 5b: rare-token bucket analysis
│       ├── 50_rerank.py                  # Phase 6: semantic reranking
│       ├── 60_backtranslate.py           # Phase 7a: classical BT from mono Shiwilu
│       ├── 60b_roundtrip_bt.py           # Phase 7a-bis: round-trip BT from Spanish
│       │                                 # (NEW; see Augmentation section)
│       ├── 61_mine_pairs.py              # Phase 7b: bitext mining via FAISS
│       ├── 62_morph_variants.py          # Phase 7c: OFF by default (needs linguist)
│       ├── 63_train_with_augmented.py    # Phase 7d: train v1_bt with augmented data
│       ├── 70_compare_runs.py            # Phase 8a: head-to-head v0/v1 pairwise comparison
│       ├── 71_human_eval_template.py     # Phase 8b: anonymized rubric CSV
│       ├── 72_leaderboard.py             # Phase 5 (closing): N-way leaderboard.{md,json}
│       ├── 73_bootstrap_ci.py            # Phase 6: percentile bootstrap CIs on chrF++
│       └── 74_thesis_tables_phase6.py    # Phase 6: LaTeX fragments for thesis tables
├── src/
│   ├── embeddings/
│   │   ├── config.py                     # SPLITS_DIR, MODELS_DIR, REPORTS_DIR, resolve_splits_dir(variant)
│   │   ├── preprocess_embeddings.py      # Train/valid/test splitter; supports --variant {main,xl}
│   │   └── exploratory/
│   │       ├── finetune_st.py            # Stage 1: train v1 sentence-transformer
│   │       ├── hard_negative_controlled.py  # Stage 2-3: mine HN + train v2/v3
│   │       ├── evaluate_retrieval.py     # Test retrieval metrics (R@K, MRR)
│   │       └── ...
│   └── nmt/
│       ├── augmentation/
│       │   ├── backtranslation.py
│       │   ├── embedding_mining.py
│       │   └── morphological_variants.py
│       ├── inference/
│       │   ├── generate.py               # generate_for_direction, predict_split, load_checkpoint
│       │   └── confidence.py
│       ├── training/
│       │   ├── model_setup.py            # Tokenizer extension + warmstart from quy/ayr/grn
│       │   ├── train_lora.py             # Trainer wiring (LoRA, DoRA, LoRA+)
│       │   └── dataset.py                # DEFAULT_WEIGHT_MAP for per-row loss weighting
│       ├── preprocessing/semantic_filter.py
│       ├── reranking/semantic_reranker.py
│       └── evaluation/
│           ├── metrics.py
│           └── rare_token.py
├── models/
│   ├── sentence_transformers/v3_iterative_hn_e5_base_bidirectional[/_xl]/  # Best embed
│   └── nmt/
│       ├── nllb_bidi_lora_v0[/_xl]/                # Baseline NMT
│       ├── nllb_bidi_lora_v1_bt[/_xl]/             # Augmented NMT (current best)
│       └── tokenizer_shw_extended/                  # NLLB tokenizer + shw_Latn
├── reports/
│   ├── 01_filtrado, 01b_unificado, 02_normalizacion, 03_auditoria/   # Data-prep reports
│   ├── 04_embeddings/{baseline,v1,v2_hn_controlled,exploratory,...}/
│   └── 05_nmt/{preprocessing[/_xl],training[/_xl],evaluation[/_xl],
│               reranking[/_xl],augmentation[/_xl]}/
└── thesis/                       # LaTeX source + auto-generated figures
```

## The "main" vs "xl" variant convention

This is **critical**. Almost every script accepts `--variant {main,xl}`
or `--splits-variant {main,xl}`. They route data through:

- `main`: original 3204-pair corpus, splits 2563/320/321
- `xl`: expanded 4501-pair corpus (after adding flashcards_oraciones,
  fidel_lomas, vs_textos_narrativos, el_principito sources), splits
  3600/450/451

`scripts/nmt/_paths.py::resolve_paths(root, variant)` is the canonical
router. It returns an `NmtPaths` dataclass with `splits_dir`,
`canonical_dir`, `filtered_dir`, `augmented_dir`, plus matching report
dirs. Suffix `_xl` is appended throughout when `variant=="xl"`.

**Default in YAMLs is `main`**. xl needs to be passed on CLI every time.

## Data pipeline (one-shot, run after raw data changes)

```powershell
# .venv-nmt for all of these
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source flashcards2
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source pdf_textos
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source flashcards_oraciones
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source fidel_lomas
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source vs_textos_narrativos
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source cotidianas
.venv-nmt/Scripts/python -m scripts.01_filtrar_dataset --source el_principito

.venv-nmt/Scripts/python -m scripts.01b_unificar_fuentes
.venv-nmt/Scripts/python -m scripts.02_depurar_dataset
.venv-nmt/Scripts/python -m scripts.03_auditar_dataset
.venv-nmt/Scripts/python -m src.embeddings.preprocess_embeddings --variant xl
```

**Bug fixed in 01b_unificar_fuentes.py**: previously dropped 4 sources because
it tried to read original column names (`TEXTO_SPA`, `español`, etc.) from the
filtered CSV, but `01_filtrar` always emits canonical `ESP`/`SHIWILU`. Fix:
when `using_filtered`, force `esp_col="ESP"`, `shi_col="SHIWILU"`,
`pair_id_col="pair_id"`. This is in `load_source()`.

## Embeddings pipeline (sentence-transformer reranker)

Always use `PYTHONPATH="src/embeddings/exploratory"` because `finetune_st.py`
and `hard_negative_controlled.py` import `evaluate_retrieval` as a sibling
module (top-level import, not relative).

```powershell
$env:PYTHONPATH = "src/embeddings/exploratory"
# Stage 1: v1 baseline fine-tune (10 epochs)
.venv-nmt/Scripts/python -m src.embeddings.exploratory.finetune_st `
    --stage v1 --model intfloat/multilingual-e5-base --epochs 10 `
    --bidirectional --splits-variant xl --output-name v1_e5_base_bidirectional_xl

# Stage 2: mine + train v2 (bumped HN)
.venv-nmt/Scripts/python -m src.embeddings.exploratory.hard_negative_controlled `
    --stage mine --model models/sentence_transformers/v1_e5_base_bidirectional_xl `
    --experiment-name v2_hn_controlled_e5_base_bidirectional_xl --bidirectional --splits-variant xl
.venv-nmt/Scripts/python -m src.embeddings.exploratory.hard_negative_controlled `
    --stage train --base-model models/sentence_transformers/v1_e5_base_bidirectional_xl `
    --experiment-name v2_hn_controlled_e5_base_bidirectional_xl `
    --epochs 2 --batch-size 32 --lr 1e-5 --bidirectional --splits-variant xl

# Stage 3: iterative HN over v2 → v3
.venv-nmt/Scripts/python -m src.embeddings.exploratory.hard_negative_controlled `
    --stage mine --model models/sentence_transformers/v2_hn_controlled_e5_base_bidirectional_xl `
    --experiment-name v3_iterative_hn_e5_base_bidirectional_xl --bidirectional --splits-variant xl
.venv-nmt/Scripts/python -m src.embeddings.exploratory.hard_negative_controlled `
    --stage train --base-model models/sentence_transformers/v2_hn_controlled_e5_base_bidirectional_xl `
    --experiment-name v3_iterative_hn_e5_base_bidirectional_xl `
    --epochs 2 --batch-size 16 --lr 5e-6 --bidirectional --splits-variant xl

# Eval (both directions)
.venv-nmt/Scripts/python -m src.embeddings.exploratory.evaluate_retrieval `
    --model models/sentence_transformers/v3_iterative_hn_e5_base_bidirectional_xl `
    --split test --tag v3_iterative_hn_e5_base_bidirectional_xl `
    --direction esp_to_shi --splits-variant xl
.venv-nmt/Scripts/python -m src.embeddings.exploratory.evaluate_retrieval `
    --model models/sentence_transformers/v3_iterative_hn_e5_base_bidirectional_xl `
    --split test --tag v3_iterative_hn_e5_base_bidirectional_xl `
    --direction shi_to_esp --splits-variant xl
```

**Bug fixed in `hard_negative_controlled.py`**: the `mine` stage was double-
appending `_xl` to the output filename when variant=xl (because
`negatives_path_for_args()` already adds it once). Fix: removed the extra
`with_name(... + "_xl.csv")` block at lines 297-300. If you re-encounter a
file at `train_hard_negatives_*_xl_xl_xl.csv`, just rename to drop one `_xl`.

### Best embeddings result (xl variant, test set, 451 pairs)

| Direction | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| ESP→SHI | 0.8160 | 0.9357 | 0.9645 | 0.8684 |
| SHI→ESP | 0.8115 | 0.9490 | 0.9645 | 0.8716 |

(vs main variant: ESP→SHI R@1 0.7882, SHI→ESP R@1 0.7913 — xl gives ~+2pp)

## NMT pipeline

### Phase 4-6 (baseline → eval → rerank)

```powershell
# Phase 4-6 prep + train v0
.venv-nmt/Scripts/python -m scripts.nmt.10_canonicalize_dataset --variant xl
.venv-nmt/Scripts/python -m scripts.nmt.20_semantic_filter --variant xl
.venv-nmt/Scripts/python -m scripts.nmt.21_build_faiss --variant xl
.venv-nmt/Scripts/python -m scripts.nmt.22_train_sentencepiece --variant xl
.venv-nmt/Scripts/python -m scripts.nmt.30_train_lora --variant xl --config config/nmt/training.yaml

# Phase 5-6 eval/rerank
.venv-nmt/Scripts/python -m scripts.nmt.40_evaluate --variant xl `
    --checkpoint models/nmt/nllb_bidi_lora_v0_xl --split test
.venv-nmt/Scripts/python -m scripts.nmt.50_rerank --variant xl `
    --checkpoint models/nmt/nllb_bidi_lora_v0_xl --split test

# Phase 8a comparison + auto-generated tables
.venv-nmt/Scripts/python -m scripts.nmt.70_compare_runs --splits-variant xl `
    --v0 nllb_bidi_lora_v0 --v1 nllb_bidi_lora_v0_xl --split test
.venv-nmt/Scripts/python -m scripts.generate_nmt_tables `
    --variant xl --v0 nllb_bidi_lora_v0 --v1 nllb_bidi_lora_v0_xl
```

**Important: `generate_nmt_tables.py` must be run as a module
(`python -m scripts.generate_nmt_tables`), NOT as a path
(`python scripts/generate_nmt_tables.py`).** Path-style fails with
`ModuleNotFoundError: No module named 'scripts.nmt'` because it imports
`scripts.nmt._paths`.

### Phase 7 (augmentation + train v1_bt)

```powershell
# 7a-bis: round-trip BT from Spanish (60b — NEW). Default Spanish source is
# OPUS-100 (FLORES needs trust_remote_code which fails on this Windows setup).
.venv-nmt/Scripts/python -m scripts.nmt.60b_roundtrip_bt --variant xl `
    --checkpoint models/nmt/nllb_bidi_lora_v0_xl `
    --source opus --n-sentences 1012 --output-name train_bt_roundtrip

# 7b: mine bitext from FAISS indexes
.venv-nmt/Scripts/python -m scripts.nmt.61_mine_pairs --variant xl

# (7c morphological variants is OFF by default — needs linguist supervision)

# 7d: train v1_bt
.venv-nmt/Scripts/python -m scripts.nmt.63_train_with_augmented --variant xl `
    --config config/nmt/training.yaml `
    --skip bt   # exclude classical Shiwilu BT (only 76 mono lines, not worth it)
```

The 60_backtranslate (classical Shiwilu BT) was **dropped** for xl runs
because the mono Shiwilu pool is only 76 lines (extracted from
`II_TEXTOS_SHIWILU.pdf`). Round-trip BT from Spanish (60b) gives ~89 high-
quality pairs at threshold 0.70 from 1012 OPUS-100 sentences — much more
useful as a starting point. Iterative BT (iter 1, iter 2) plans to scale this
up with Wikipedia + Tatoeba + News-commentary.

### Phase 5/6 closing artifacts (cycle is finished)

```powershell
# Phase 5: unified leaderboard across all 6 trained runs
.venv-nmt/Scripts/python -m scripts.nmt.72_leaderboard --variant xl --split test
# → reports/05_nmt/evaluation_xl/leaderboard.{md,json}

# Phase 6: bootstrap 95% CIs (n=1000) on chrF++ for each run
.venv-nmt/Scripts/python -m scripts.nmt.73_bootstrap_ci --variant xl --n-boot 1000
# → reports/05_nmt/evaluation_xl/<run>/bootstrap_ci.json
# → reports/05_nmt/evaluation_xl/bootstrap_ci_summary.md

# Phase 6: regenerate thesis LaTeX fragments (leaderboard + CI + rare-token full)
.venv-nmt/Scripts/python -m scripts.nmt.74_thesis_tables_phase6 --variant xl
# → thesis/latex/figuras/generated/nmt_{leaderboard,bootstrap_ci,rare_token_full}_xl.tex

# Rebuild thesis PDF (xelatex + biber via latexmk, ~1 min)
cd thesis/latex && latexmk -xelatex -interaction=nonstopmode -outdir=build tesis.tex
# → thesis/latex/build/tesis.pdf (also copied to pdf/tesis.pdf)
```

### Final NMT leaderboard — shipped (xl test set, 892 directional rows, reranked)

| Run | shw→spa chrF / BLEU | spa→shw chrF / BLEU | avg chrF | 95% CI on avg |
|---|---|---|---:|---|
| v0_xl | 25.16 / 9.73 | 28.06 / 4.61 | 26.61 | [25.33, 27.88] |
| v1_bt_xl (+BT OPUS-100) | 29.91 / 12.45 | 34.36 / 6.69 | 32.14 | [30.63, 33.80] |
| v2.0 DoRA | 30.82 / 14.55 | 34.35 / 6.46 | 32.58 | [31.02, 34.25] |
| v2.1 DoRA+LoRA+ | 41.05 / 23.23 | 47.54 / 12.76 | 44.30 | [42.49, 46.12] |
| **v2.1b LoRA+ ★ champion** | **42.42 / 24.48** | **47.56 / 12.45** | **44.99** | **[43.17, 46.96]** |
| v2.2 BT iter1 Wikipedia | 30.72 / 14.76 | 33.94 / 5.90 | 32.33 | [30.82, 33.94] |

Best alpha for the v2.1b reranker = 0.7. Two clean statistical tiers
(non-overlapping 95% CIs between the two clusters), and within-tier
ranking (e.g. v2.1b vs v2.1, or v2.0 vs v1_bt) is NOT statistically
distinguishable — report honestly that LoRA+ alone is the win, not
"LoRA+ minus DoRA".

## Training internals — what's already on by default

These are NOT new tricks to add; they are already wired into the baseline:

- **Warmstart embeddings**: new `shw_Latn` token embeddings are mean-init from
  `quy_Latn + ayr_Latn + grn_Latn` (Andean indigenous neighbors). See
  `model_setup.py:114-135` and `training.yaml:10-13`. **Do not propose
  "adding warmstart from Quechua" as an improvement** — already there.
- **Per-row loss weighting (Enhancement #4)**: synthetic origins (`mined_v3_sbert`,
  `backtranslation_v0`, `backtranslation_roundtrip_v0`) get weight <1.0,
  real parallel keeps weight 1.0. See `dataset.py::DEFAULT_WEIGHT_MAP`.
  Disable with `--no-weighting`.
- **LoRA bump for v1_bt**: `r=32, alpha=64` (vs v0's `r=16, alpha=32`). Set
  via `V1_BT_LORA` constant in `63_train_with_augmented.py`. Disable with
  `--no-lora-bump`.
- **Bidirectional**: training data has both spa→shw and shw→spa rows; one
  LoRA adapter handles both.
- **Intermediate eval is lite**: `make_metrics_fn` only computes BLEU + chrF
  via sacrebleu during training-loop eval. COMET + BERTScore only run in
  `40_evaluate.py` for final test eval. So **"skip COMET in intermediate"
  is not a speedup to add** — already done.

## Phase 0 ablation flags (added during current session)

`30_train_lora.py` and `63_train_with_augmented.py` both expose these:

| Flag | Default | What it does |
|---|---|---|
| `--use-dora` | off | Switch from LoRA to DoRA (Decomposed LoRA). PEFT 0.16+ native. |
| `--loraplus-lr-ratio R` | 0.0 | LoRA+ asymmetric LR (lr_B = R × lr_A). 16.0 is paper default. Wires `peft.optimizers.create_loraplus_optimizer`. |
| `--rank N` | yaml | Override LoRA r |
| `--alpha N` | yaml | Override LoRA alpha |
| `--bf16` | off (auto picks fp16) | Use bf16 instead of fp16 (more numerically stable) |
| `--compile` | off | **DOES NOT WORK ON WINDOWS** (Triton missing). Don't pass. |
| `--direction {shw2spa,spa2shw}` | None | Filter dataset to one direction. Used for Two-DoRA training (one adapter per direction). |
| `--output-dir PATH` (30 only) / `--output PATH` (63 only) | yaml | Override checkpoint output dir |
| `--run-suffix STR` (30 only) | None | Append to default run name |

`40_evaluate.py` got Two-DoRA support: `--checkpoint-spa2shw` +
`--checkpoint-shw2spa` + `--run-name`. When provided, it loads each
adapter in turn for its respective direction. `50_rerank.py` accepts
`--run-name` (instead of requiring `--checkpoint`) for the same case.

## Common bugs / gotchas (solved or documented)

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'datasets'` | Using `.venv` | Switch to `.venv-nmt` |
| `cuda: False` | Using `.venv` (CPU-only torch) | Switch to `.venv-nmt` |
| `from evaluate_retrieval import ...` ImportError | `python -m` doesn't add sibling dir to sys.path | `$env:PYTHONPATH = "src/embeddings/exploratory"` before invoking |
| HN train can't find mined CSV | mine stage double-appended `_xl` | Patched in `hard_negative_controlled.py:296-299`. Old files: rename to drop one `_xl`. |
| 01b drops 4 sources silently | filtered CSVs always have ESP/SHIWILU but config says original column names | Patched in `01b_unificar_fuentes.py::load_source` |
| `TritonMissing` from torch.compile | Triton not on Windows | Don't use `--compile` flag |
| `Dataset 'openlanguagedata/flores_plus' is gated` | HF auth needed | Use `gsarti/flores_101` or OPUS-100 fallback (already wired in 60b) |
| `An error occurred while generating the dataset` for `Muennighoff/flores200` | Scripted dataset borked on Windows | Use OPUS-100 fallback (auto in 60b) |
| `generate_nmt_tables.py: ModuleNotFoundError: scripts.nmt` | Run as path-style instead of `-m` | Always use `.venv-nmt/Scripts/python -m scripts.generate_nmt_tables ...` |
| `NameError: name 'bf16' is not defined` in info dict | `resolve_precision` was unpacked as `fp16, _` | Patched: `fp16, bf16 = resolve_precision(...)` in `train_lora.py` |

## Why spa→shw BLEU is so much lower than shw→spa (defend in thesis)

This is a structural pattern, not a bug. Four reasons:

1. **NLLB decoder bias**: NLLB-200 was pre-trained on billions of Spanish
   tokens, near-zero Shiwilu. Generating fluent Shiwilu requires the model
   to PRODUCE in a language it barely knew before LoRA. Generating Spanish
   is closer to what it already does.
2. **Morphological asymmetry**: Shiwilu is agglutinative. Each output word
   needs many decisions (person, aspect, evidentiality, applicative, ...).
   Spanish output is far less morphologically dense.
3. **New embeddings, undertrained**: The `shw_Latn` token embeddings were
   added fresh and trained on only ~6.9k pairs.
4. **Single-reference BLEU**: Multiple Shiwilu translations are valid for
   the same Spanish. Single-ref BLEU undercounts.

**Look at chrF/BLEU ratio**: shw→spa = 2.94, spa→shw = 6.88. The model is
producing right morpheme stems but failing at full wordforms. **Defend
chrF++ as primary metric in thesis** — and note that for v1_bt_xl reranked,
spa→shw chrF (33.85) is actually HIGHER than shw→spa chrF (31.20).

## Conventions for future Claude sessions

1. **Always invoke as module** (`python -m scripts.nmt.X`), never as path.
2. **Always pass `--variant xl`** unless explicitly working on the small
   `main` corpus.
3. **Always use `.venv-nmt/Scripts/python`** unless you have a specific
   reason to use `.venv`.
4. **Never propose "warmstart embeddings from Quechua"** as a future
   improvement — already on. Same for "skip COMET in intermediate eval"
   and "use direction-weighted loss" (the user explicitly rejected the
   latter, preferring Two DoRAs as the architectural fix).
5. **Never use `torch.compile` on this Windows machine.**
6. **Never amend commits** unless explicitly asked. Never push without
   explicit approval. Never use `--no-verify`.
7. **Default to chrF++ for spa→shw discussion**, BLEU is misleading.
8. **Background long training jobs** with `run_in_background=True` and
   monitor via task notifications rather than polling. Each NMT train run
   is ~2-3.5h on the xl dataset.
9. **Log everything to `logs/pipeline_stepXX_*.log`** with consistent
   naming so the user can grep history.

## Future-work backlog

Phase 2 ablations are **closed**. Final reranked-test leaderboard (xl, 892 rows):

| Run | shw→spa chrF | spa→shw chrF | avg chrF |
|---|---|---|---|
| v0_xl | 24.68 | 26.20 | 25.44 |
| v1_bt_xl (reranked) | 31.20 | 33.85 | 32.52 |
| v2.0 DoRA | 30.82 | 34.35 | 32.58 |
| **v2.1b LoRA+** ← champion | 42.42 | 47.56 | **44.99** |
| v2.1 DoRA+LoRA+ | 41.05 | 47.54 | 44.30 |
| v2.2 BT iter 1 (Wikipedia es) | 30.72 | 33.94 | 32.33 (regression) |

**Key takeaways (already validated — do not redo):**
- **LoRA+ alone is the win** (+12.47 avg chrF over v1_bt_xl). DoRA does
  not pay for itself at this corpus size; pure LoRA+ even slightly beats
  DoRA+LoRA+.
- **Wikipedia-sourced BT actively hurts** — too far from the gold
  conversational/narrative domain. The successful BT (v1_bt_xl) used
  OPUS-100 (conversational web corpus) with 1012 source sentences and
  only 89 high-quality pairs at threshold 0.70. **Lesson: domain match
  beats volume in low-resource BT.**

**Deferred to future work** (code is wired, skipped in this thesis cycle):

- **Two-Adapter LoRA+** (asymmetric ranks). One LoRA+ adapter per
  direction: spa→shw r=64 α=128, shw→spa r=32 α=64. Use `--direction
  {spa2shw,shw2spa}` filter in `30_train_lora.py`. Combine at eval via
  `40_evaluate.py --checkpoint-spa2shw <path> --checkpoint-shw2spa
  <path> --run-name nllb_two_loraplus_xl`. Hypothesis to validate next
  cycle: separating directions closes the residual gap. ~5–6h total.
- **SWA (Stochastic Weight Averaging)** over the last 3–5 checkpoints
  of the Phase 3 winner. Cheap (~30 min), typically +0.3–0.8 chrF in
  low-resource regimes (Izmailov et al., 2018). Not yet implemented as
  a script; would live as `scripts/nmt/35_swa_average.py`.
- **In-domain iterative BT iter 2**. Skip Wikipedia. Curate Tatoeba +
  News-commentary + community/oral-narrative corpora, semantic-filter
  against the v3 SBERT, train iter 2 with LoRA+.

The user prefers **architectural solutions over loss-weighting hacks** —
they explicitly rejected direction-weighted loss in favor of Two-Adapter.

**This thesis cycle closes on v2.1b LoRA+ as the shipped champion.**
Remaining work: Phase 5 (full eval suite — BLEU + chrF++ + BERTScore +
COMET on all survivors) and Phase 6 (bootstrap CIs + rare-token + morph
buckets + thesis tables/PDF rebuild).

## Where to look first when something breaks

1. **Read the failing log**: `logs/pipeline_stepXX_*.log` (tail or grep).
2. **Check the env**: `which python`, `torch.cuda.is_available()`. If wrong,
   you're using `.venv` instead of `.venv-nmt`.
3. **Check the variant flag**: missing `--variant xl` is the most common
   silent failure (model trains on the wrong dataset).
4. **Check `_paths.py`**: it's the source of truth for variant routing.
5. **Look at this file's "Common bugs" section** for known fixes.
