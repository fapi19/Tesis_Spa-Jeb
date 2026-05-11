# Protocolo de Validacion Humana

Este documento resume el protocolo de evaluacion humana preparado para el sistema NMT.
Documenta la muestra, rubrica, anonimizacion y comando reproducible; no contiene puntajes humanos.

## Alcance

- Objetivo: complementar los resultados automaticos del NMT con una revision participativa posterior por hablantes de shiwilu o revisores competentes.
- Estado: protocolo preparado, no ejecutado con revisores.
- Por ello, no se reportan promedios humanos, acuerdo interevaluador ni analisis cualitativo de respuestas.

## Comando reproducible

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.71_human_eval_template --variant xl --per-direction 100 --seed 2026 --split test
```

## Salidas

| Campo | Valor |
|---|---|
| Plantilla CSV | `reports/05_nmt/evaluation_xl/human_eval_template.csv` |
| Clave anonima | `reports/05_nmt/evaluation_xl/human_eval_anon_key.json` |
| Protocolo Markdown | `reports/05_nmt/evaluation_xl/human_eval_protocol.md` |

La clave anonima debe mantenerse separada de los revisores. Los revisores solo deben recibir el CSV o un formulario derivado de el.

## Muestra

| Campo | Valor |
|---|---|
| Generado en UTC | 2026-05-11T19:09:07.394317+00:00 |
| Variante | `xl` |
| Split | `test` |
| Items solicitados por direccion | 100 |
| Filas generadas | 200 |
| Direcciones | `shw2spa`=100, `spa2shw`=100 |
| Estratificacion | `origin_source` y bucket de longitud fuente: short <= 5, medium <= 12, long > 12 palabras |

### Distribucion por origen

| origin_source | rows |
|---|---:|
| `fidel_lomas` | 23 |
| `flashcards2` | 108 |
| `flashcards_oraciones` | 28 |
| `pdf_textos` | 41 |

## Sistemas comparados

| Columna anonima | Sistema fuente | Predicciones disponibles |
|---|---|---:|
| `hypothesis_A` | oculto para el revisor | True |
| `hypothesis_B` | oculto para el revisor | True |
| `hypothesis_C` | oculto para el revisor | True |
| `hypothesis_D` | oculto para el revisor | True |

El mapeo oculto letra-sistema se guarda solo en el JSON de clave anonima.
Los sistemas comparados son v0, v0 reranked, v1_bt y v1_bt reranked.

## Rubrica

| Dimension | Escala | Criterio |
|---|---|---|
| adequacy_1_5 | 1-5 | Preservacion del sentido; penaliza omisiones, agregados y cambios semanticos. |
| fluency_1_5 | 1-5 | Gramaticalidad, naturalidad y legibilidad en la lengua destino. |
| cultural_relevance_1_5 | 1-5 | Registro idiomatico y elecciones lexicas culturalmente apropiadas. |
| notes | texto libre | Explicacion opcional de errores, dudas o casos culturalmente marcados. |

## Instrucciones para revisores

1. Leer la fuente y la referencia.
2. Puntuar cada hipotesis anonimizada de forma independiente en adecuacion, fluidez y pertinencia cultural.
3. Usar enteros de 1 a 5; dejar una nota cuando una baja puntuacion dependa de registro cultural, ambiguedad o falta de contexto.
4. No intentar inferir que sistema produjo cada hipotesis.

## Columnas de la plantilla

`id`, `pair_id`, `direction`, `origin_source`, `source`, `reference`, `hypothesis_A`, `hypothesis_B`, `hypothesis_C`, `hypothesis_D`, `adequacy_1_5`, `fluency_1_5`, `cultural_relevance_1_5`, `notes`

Columnas de hipotesis preparadas: `hypothesis_A`, `hypothesis_B`, `hypothesis_C`, `hypothesis_D`
