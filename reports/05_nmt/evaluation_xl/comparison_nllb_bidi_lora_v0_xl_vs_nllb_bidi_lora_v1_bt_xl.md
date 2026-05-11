# SA-BiNLLB run comparison (nllb_bidi_lora_v0_xl vs nllb_bidi_lora_v1_bt_xl)

- Split: `test`
- Variant: `xl`
- Timestamp UTC: 2026-05-10T00:31:08.346601+00:00
- Headline metric: chrF++ (per plan section 31)

## Shiwilu -> Spanish (`shw2spa`)

| Variant | chrF++ | BLEU | BERTScore F1 | COMET |
|---|---|---|---|---|
| nllb_bidi_lora_v0_xl | 22.84 | 6.74 | 0.90 | 0.60 |
| nllb_bidi_lora_v0_xl + reranker | 24.68 | 8.40 | 0.91 | 0.61 |
| nllb_bidi_lora_v1_bt_xl | 30.07 | 14.01 | 0.91 | 0.65 |
| nllb_bidi_lora_v1_bt_xl + reranker | 31.20 | 14.91 | 0.91 | 0.66 |

### Rare-token / OOV breakdown (>=20% rare bucket)

| Variant | chrF++ rare | OOV recovery (rec/tot) |
|---|---|---|
| nllb_bidi_lora_v0_xl | - | - (-/-) |
| nllb_bidi_lora_v0_xl + reranker | - | - (-/-) |
| nllb_bidi_lora_v1_bt_xl | - | - (-/-) |
| nllb_bidi_lora_v1_bt_xl + reranker | - | - (-/-) |

## Spanish -> Shiwilu (`spa2shw`)

| Variant | chrF++ | BLEU | BERTScore F1 | COMET |
|---|---|---|---|---|
| nllb_bidi_lora_v0_xl | 24.82 | 3.65 | 0.87 | 0.66 |
| nllb_bidi_lora_v0_xl + reranker | 26.20 | 3.81 | 0.88 | 0.67 |
| nllb_bidi_lora_v1_bt_xl | 32.76 | 5.50 | 0.89 | 0.71 |
| nllb_bidi_lora_v1_bt_xl + reranker | 33.85 | 5.20 | 0.89 | 0.72 |

### Rare-token / OOV breakdown (>=20% rare bucket)

| Variant | chrF++ rare | OOV recovery (rec/tot) |
|---|---|---|
| nllb_bidi_lora_v0_xl | - | - (-/-) |
| nllb_bidi_lora_v0_xl + reranker | - | - (-/-) |
| nllb_bidi_lora_v1_bt_xl | - | - (-/-) |
| nllb_bidi_lora_v1_bt_xl + reranker | - | - (-/-) |

## Direction-averaged headline numbers

| Variant | avg chrF++ | avg BLEU | avg chrF++ rare | avg OOV recovery |
|---|---|---|---|---|
| nllb_bidi_lora_v0_xl | 23.83 | 5.20 | - | - |
| nllb_bidi_lora_v0_xl + reranker | 25.44 | 6.11 | - | - |
| nllb_bidi_lora_v1_bt_xl | 31.41 | 9.75 | - | - |
| nllb_bidi_lora_v1_bt_xl + reranker | 32.52 | 10.05 | - | - |

## Caveats

- BERTScore is computed with `xlm-roberta-large`. Shiwilu is OOD for the underlying encoder; treat Shiwilu-side BERTScore as proxy.
- COMET (`Unbabel/wmt22-comet-da`) was not trained on Shiwilu data; report as indicative, not absolute.
- chrF++ remains the primary metric for low-resource, morphology-rich languages.

## Source files
- `reports/05_nmt/evaluation/<run>/<split>_metrics.json`
- `reports/05_nmt/reranking/<run>/<split>_metrics_reranked.json`