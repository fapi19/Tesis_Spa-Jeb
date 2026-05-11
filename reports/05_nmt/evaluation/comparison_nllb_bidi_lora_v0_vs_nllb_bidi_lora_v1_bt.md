# SA-BiNLLB run comparison (nllb_bidi_lora_v0 vs nllb_bidi_lora_v1_bt)

- Split: `test`
- Timestamp UTC: 2026-05-09T16:56:00.359053+00:00
- Headline metric: chrF++ (per plan section 31)

## Shiwilu -> Spanish (`shw2spa`)

| Variant | chrF++ | BLEU | BERTScore F1 | COMET |
|---|---|---|---|---|
| nllb_bidi_lora_v0 | 18.63 | 3.49 | 0.89 | 0.60 |
| nllb_bidi_lora_v0 + reranker | 19.95 | 3.71 | 0.90 | 0.61 |
| nllb_bidi_lora_v1_bt | - | - | - | - |
| nllb_bidi_lora_v1_bt + reranker | - | - | - | - |

### Rare-token / OOV breakdown (>=20% rare bucket)

| Variant | chrF++ rare | OOV recovery (rec/tot) |
|---|---|---|
| nllb_bidi_lora_v0 | 18.66 | 0.030 (11/365) |
| nllb_bidi_lora_v0 + reranker | 19.86 | 0.038 (14/365) |
| nllb_bidi_lora_v1_bt | - | - (-/-) |
| nllb_bidi_lora_v1_bt + reranker | - | - (-/-) |

## Spanish -> Shiwilu (`spa2shw`)

| Variant | chrF++ | BLEU | BERTScore F1 | COMET |
|---|---|---|---|---|
| nllb_bidi_lora_v0 | 20.65 | 2.03 | 0.87 | 0.65 |
| nllb_bidi_lora_v0 + reranker | 21.84 | 2.21 | 0.87 | 0.67 |
| nllb_bidi_lora_v1_bt | - | - | - | - |
| nllb_bidi_lora_v1_bt + reranker | - | - | - | - |

### Rare-token / OOV breakdown (>=20% rare bucket)

| Variant | chrF++ rare | OOV recovery (rec/tot) |
|---|---|---|
| nllb_bidi_lora_v0 | 20.51 | 0.014 (6/428) |
| nllb_bidi_lora_v0 + reranker | 21.64 | 0.014 (6/428) |
| nllb_bidi_lora_v1_bt | - | - (-/-) |
| nllb_bidi_lora_v1_bt + reranker | - | - (-/-) |

## Direction-averaged headline numbers

| Variant | avg chrF++ | avg BLEU | avg chrF++ rare | avg OOV recovery |
|---|---|---|---|---|
| nllb_bidi_lora_v0 | 19.64 | 2.76 | 19.58 | 0.022 |
| nllb_bidi_lora_v0 + reranker | 20.89 | 2.96 | 20.75 | 0.026 |
| nllb_bidi_lora_v1_bt | - | - | - | - |
| nllb_bidi_lora_v1_bt + reranker | - | - | - | - |

## Caveats

- BERTScore is computed with `xlm-roberta-large`. Shiwilu is OOD for the underlying encoder; treat Shiwilu-side BERTScore as proxy.
- COMET (`Unbabel/wmt22-comet-da`) was not trained on Shiwilu data; report as indicative, not absolute.
- chrF++ remains the primary metric for low-resource, morphology-rich languages.

## Source files
- `reports/05_nmt/evaluation/<run>/<split>_metrics.json`
- `reports/05_nmt/reranking/<run>/<split>_metrics_reranked.json`