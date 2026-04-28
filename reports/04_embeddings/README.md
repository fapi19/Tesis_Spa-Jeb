# Reports 04 Embeddings

Esta carpeta está organizada por etapa/experimento para evitar mezclar reportes.

## Estructura

- `preprocessing/`: manifiesto, auditoría y cierre del preprocesamiento canónico.
- `baseline/`: evaluación del modelo `intfloat/multilingual-e5-small` sin fine-tuning.
- `v1/`: fine-tuning E5 con `MultipleNegativesRankingLoss`.
- `controlled_hn/`: minería, muestras y validación de hard/medium negatives.
- `v2_hn_controlled/`: modelo candidato actual con hard + medium negatives.
- `v2_hn_controlled_hard/`: ablación usando solo hard negatives.
- `legacy_v2/`: resultados antiguos con triplets/hard negatives no controlados.
- `exploratory/`: análisis exploratorios previos como similarity scores y comparaciones.

## Modelo Candidato Actual

El candidato actual es `v2_hn_controlled`.

Reporte principal:

- `v2_hn_controlled/v2_hn_controlled_retrieval.json`
- `v2_hn_controlled/v2_hn_controlled_training.json`
- `v2_hn_controlled/v2_hn_controlled_freeze_metadata.json`

## Nota

`v1` se mantiene como baseline fuerte. `legacy_v2` no debe usarse como modelo principal.
