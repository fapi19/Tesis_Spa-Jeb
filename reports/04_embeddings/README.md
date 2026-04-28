# Reports 04 Embeddings

Esta carpeta está organizada por etapa/experimento para evitar mezclar reportes.

## Estructura

- `preprocessing/`: manifiesto, auditoría y cierre del preprocesamiento canónico.
- `baseline/`: evaluación del modelo `intfloat/multilingual-e5-small` sin fine-tuning.
- `v1/`: fine-tuning E5 con `MultipleNegativesRankingLoss`.
- `controlled_hn/`: minería, muestras y validación de hard/medium negatives.
- `v2_hn_controlled/`: candidato anterior con E5-small + hard/medium negatives.
- `v2_hn_controlled_e5_base/`: candidato anterior con E5-base + hard/medium negatives.
- `v2_hn_controlled_e5_base_bidirectional/`: candidato anterior bidireccional.
- `v3_iterative_hn_e5_base_bidirectional/`: entrenamiento del candidato actual.
- `v2_hn_controlled_hard/`: ablación usando solo hard negatives.
- `experiments/`: evaluaciones nuevas como `baseline_e5_base`, `v1_e5_base` y
  `v3_iterative_hn_e5_base_bidirectional`.
- `legacy_v2/`: resultados antiguos con triplets/hard negatives no controlados.
- `exploratory/`: análisis exploratorios previos como similarity scores y comparaciones.

## Modelo Candidato Actual

El candidato actual es `v3_iterative_hn_e5_base_bidirectional`.

Métricas principales sobre 321 pares de test:

- Español -> Shiwlu: `R@1=0.7882`, equivalente a 253 aciertos en rank 1;
  `R@5=0.9283`, `R@10=0.9782`, `MRR=0.8480`.
- Shiwlu -> español: `R@1=0.7913`, equivalente a 254 aciertos en rank 1;
  `R@5=0.9564`, `R@10=0.9688`, `MRR=0.8617`.
- Mejora frente a `v2_hn_controlled_e5_base_bidirectional`: `+3.74` puntos
  porcentuales en `R@1` español -> Shiwlu (`+4.98%` relativo) y `+1.25`
  puntos porcentuales en `R@1` Shiwlu -> español (`+1.60%` relativo).

Reporte principal:

- `experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_esp_to_shi_retrieval.json`
- `experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_shi_to_esp_retrieval.json`
- `experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_esp_to_shi_r1_error_analysis_summary.json`
- `experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_shi_to_esp_r1_error_analysis_summary.json`
- `experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_freeze_metadata.json`
- `v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_training.json`

## Nota

`v2_hn_controlled_e5_base_bidirectional` queda como candidato anterior fuerte.
`legacy_v2` no debe usarse como modelo principal.

La iteración v3 mejora `R@1` y `MRR` en ambos sentidos frente al candidato
anterior y pasa los criterios de aceptación. La fase de embeddings queda cerrada
provisionalmente; la siguiente fase es usar `v3_iterative_hn_e5_base_bidirectional`
en integración/evaluación con NMT.

## Resumen Final

- Candidato final provisional: `v3_iterative_hn_e5_base_bidirectional`.
- Razón: mejor `R@1` y `MRR` bidireccional.
- Riesgo: `R@10` Shiwlu -> español baja `-0.31` puntos porcentuales.
- Decisión: no hacer más minería iterativa salvo evidencia cualitativa fuerte.
- Siguiente paso: integración/evaluación con NMT.
