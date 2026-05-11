# Bootstrap 95% CIs on chrF++ — Phase 6

Generated 2026-05-11T14:53:57+00:00. Bootstrap n=1000 resamples per run. Predictions are **reranked** unless flagged otherwise.

| Run | shw→spa chrF++ [95% CI] | spa→shw chrF++ [95% CI] | avg chrF++ [95% CI] |
|---|---|---|---|
| v0_xl (reranked) | 25.16 [23.07, 27.47] | 28.06 [26.56, 29.74] | 26.61 [25.33, 27.88] |
| v1_bt_xl (reranked) | 29.91 [27.55, 32.35] | 34.36 [32.42, 36.69] | 32.14 [30.63, 33.80] |
| v2.0 DoRA (reranked) | 30.82 [28.44, 33.12] | 34.35 [32.54, 36.59] | 32.58 [31.02, 34.25] |
| v2.1 DoRA+LoRA+ (reranked) | 41.05 [38.34, 43.56] | 47.54 [45.36, 49.95] | 44.30 [42.49, 46.12] |
| v2.1b LoRA+ (reranked) | 42.42 [39.65, 44.94] | 47.56 [45.02, 50.27] | 44.99 [43.17, 46.96] |
| v2.2 BT iter1 (reranked) | 30.72 [28.41, 33.07] | 33.94 [31.87, 36.26] | 32.33 [30.82, 33.94] |

Notes:
- CI = percentile bootstrap on the 446 test items per direction (892 total).
- `avg` CI is computed by independently resampling each direction and averaging — it is wider than a fixed-weight transform of the per-direction CIs.
- Overlapping 95% CIs between two runs *do not* prove the diff is non-significant, but non-overlapping CIs do strongly suggest a real difference.
