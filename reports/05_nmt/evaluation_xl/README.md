# NMT Evaluation XL Artifacts

This directory stores the closed automatic-evaluation reports and the prepared
human-evaluation protocol for the `xl` variant of the Shiwilu-Spanish NMT
system.

## Automatic Evaluation

| File | Purpose |
|---|---|
| `leaderboard.md` | Six-run test leaderboard with baseline and reranked metrics. |
| `leaderboard.json` | Machine-readable version of the leaderboard. |
| `bootstrap_ci_summary.md` | Bootstrap 95% confidence intervals for reranked chrF++. |
| `<run>/test_predictions.jsonl` | Test predictions for each run. |
| `<run>/bootstrap_ci.json` | Per-run bootstrap details when available. |

The thesis uses chrF++ as the primary automatic metric and reports BLEU/COMET
as complementary signals. COMET is treated as indicative because it was not
trained on Shiwilu. Per the MT-evaluation expert's advice, each direction is
also read with its most informative headline metric: BLEU for `shw2spa`
(Spanish output) and chrF++ for `spa2shw` (Shiwilu output).

## Stratified Qualitative Analysis

| File | Purpose |
|---|---|
| `<run>/qualitative/bucket_summary.json` | Per-sentence score distribution by direction; buckets `shw2spa` by BLEU (<10 / 10-20 / >20) and `spa2shw` by chrF++ (<20 / 20-40 / >40). |
| `<run>/qualitative/sampled_examples.csv` | Seeded stratified sample of source/reference/hypothesis per bucket. |
| `<run>/qualitative/qualitative_report.md` | Human-readable narrative used in thesis section `nmt-cualitativo`. |

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.42_qualitative_analysis --variant xl
```

## Pairwise Forced-Choice Preference

| File | Purpose |
|---|---|
| `pairwise_preference.xlsx` | Blind A/B workbook: reference + two anonymized system outputs, randomized order, ties allowed. Default `shw2spa`, 60 pairs. |
| `pairwise_preference_anon_key.json` | Per-row A/B-to-system mapping. Keep private; do not distribute to reviewers. |

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.76_pairwise_preference --variant xl
```

By default it compares `v2.1b` (champion) vs `v2.1` (DoRA+LoRA+) to resolve the
statistically tied selection. Override with `--run-a` / `--run-b`.

## Human-Evaluation Protocol

| File | Purpose |
|---|---|
| `human_eval_template.csv` | Reviewer template with 100 items per direction, four anonymized hypotheses and empty rubric fields. |
| `human_eval_anon_key.json` | Secret A/B/C/D mapping from anonymous labels to systems. Do not distribute to reviewers. |
| `human_eval_protocol.md` | Public protocol summary: scope, sample, rubric, command and reviewer instructions. |

The protocol is prepared but not executed. Therefore this directory intentionally
does not contain human averages, inter-annotator agreement, reviewer comments or
expert actas.

## Reproducible Command

Run from the repository root:

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.71_human_eval_template --variant xl --per-direction 100 --seed 2026 --split test
```

By default the script writes CSV, JSON and Markdown outputs. To skip the
Markdown summary:

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.71_human_eval_template --variant xl --per-direction 100 --seed 2026 --split test --no-write-md
```

To write the Markdown summary elsewhere:

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.71_human_eval_template --variant xl --per-direction 100 --seed 2026 --split test --md-output path/to/protocol.md
```
