# NMT Leaderboard — Phase 5 (split=test, variant=xl)

Generated 2026-05-11T14:32:18Z. All metrics on the held-out **test** set (446 pairs × 2 directions = 892 rows).

## Baseline (no reranking)

| Run | shw→spa chrF++ / BLEU / BERTScore-F1 / COMET | spa→shw chrF++ / BLEU / BERTScore-F1 / COMET | avg chrF++ |
|---|---|---|---:|
| v0_xl (baseline LoRA r=16) | 23.58 / 8.50 / 0.903 / 0.602 | 26.44 / 4.47 / 0.873 / 0.664 | **25.01** |
| v1_bt_xl (+BT OPUS-100, LoRA r=32) | 29.01 / 12.03 / 0.909 / 0.636 | 32.69 / 5.57 / 0.884 / 0.700 | **30.85** |
| v2.0 DoRA | 29.37 / 13.62 / 0.910 / 0.636 | 33.30 / 5.79 / 0.889 / 0.713 | **31.34** |
| v2.1 DoRA + LoRA+ | 40.68 / 23.48 / 0.921 / 0.697 | 46.96 / 11.88 / 0.911 / 0.779 | **43.82** |
| v2.1b LoRA+ (champion) | 40.34 / 22.80 / 0.922 / 0.703 | 46.74 / 12.60 / 0.911 / 0.778 | **43.54** |
| v2.2 +BT iter1 Wikipedia (regression) | 29.67 / 14.11 / 0.912 / 0.644 | 32.48 / 5.83 / 0.885 / 0.706 | **31.08** |

## Reranked (best-alpha by avg chrF++)

| Run | best α | shw→spa chrF++ / BLEU | spa→shw chrF++ / BLEU | avg chrF++ | Δ vs baseline |
|---|---:|---|---|---:|---:|
| v0_xl (baseline LoRA r=16) | 0.70 | 25.16 / 9.73 | 28.06 / 4.61 | **26.61** | +1.60 |
| v1_bt_xl (+BT OPUS-100, LoRA r=32) | 0.30 | 29.91 / 12.45 | 34.36 / 6.69 | **32.14** | +1.28 |
| v2.0 DoRA | 0.70 | 30.82 / 14.55 | 34.35 / 6.46 | **32.58** | +1.25 |
| v2.1 DoRA + LoRA+ | 0.70 | 41.05 / 23.23 | 47.54 / 12.76 | **44.30** | +0.47 |
| v2.1b LoRA+ (champion) | 0.70 | 42.42 / 24.48 | 47.56 / 12.45 | **44.99** | +1.45 |
| v2.2 +BT iter1 Wikipedia (regression) | 0.30 | 30.72 / 14.76 | 33.94 / 5.90 | **32.33** | +1.25 |

---

**Shipped champion: `v2.1b LoRA+ (champion)`** with reranked avg chrF++ = **44.99** (α = 0.70).

Caveats (from each `test_metrics.json`):
- **BERTScore** uses `xlm-roberta-large` which has multilingual coverage but Shiwilu is OOD. Treat Shiwilu-side BERTScore as proxy only.
- **COMET** (`Unbabel/wmt22-comet-da`) was not trained on Shiwilu. Reported as indicative only.
- Primary headline metric is **chrF++** per plan §31.
