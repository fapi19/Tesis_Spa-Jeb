# Protocolo de validación humana

Este documento resume el protocolo de validación humana del sistema NMT y su primera ejecución participativa.

## Alcance

- Objetivo: complementar los resultados automáticos del NMT con revisión participativa por hablantes de shiwilu o revisores competentes.
- Estado: protocolo documentado y primera ejecución completada.
- Revisor de la primera ejecución: Fidel Lomas Chota, hablante/revisor competente y uno de los principales exponentes y preservadores contemporáneos del shiwilu vinculados al proyecto.
- Muestra revisada: 100 ítems, 50 shiwilu->castellano y 50 castellano->shiwilu.
- Archivo completo revisado: `reports/05_nmt/evaluation_xl/validacion_participativa_100.xlsx`.
- Enlace externo de consulta: https://docs.google.com/spreadsheets/d/1dYJ_4qstg76dHtDs6qaIICG5aBeiG7kL/edit?usp=sharing&ouid=110915706047387486036&rtpof=true&sd=true
- Constancia oral del revisor: https://drive.google.com/file/d/1vmpMNzfUMuKzzUV1f3pqhYOJWmvLWiV1/view?usp=sharing
- Registro complementario de interaccion: `thesis/latex/anexos_digitales/Anexo_E_registro_interaccion_validacion.png` / https://drive.google.com/file/d/1OBcgrEWWMM6FFewm5_iqtCb2o9z1tIZV/view?usp=sharing
- Constancia visual de conformidad: `thesis/latex/anexos_digitales/Anexo_E_constancia_conformidad_validacion.png` / https://drive.google.com/file/d/1gN4nYax5ZVNwlFRW_ZjlZgUSPKMlZAMj/view?usp=sharing
- Nota: estos registros son respaldos complementarios; no se reproducen como figuras principales por contener elementos de comunicacion privada.

La primera ejecución aporta evidencia humana trazable y especialmente valiosa por la trayectoria del revisor. El instrumento puede repetirse con más hablantes si la disponibilidad lo permite; en ese caso, conviene reutilizar la misma muestra y rúbrica para calcular acuerdo interevaluador. Esta ampliación se plantea como fortalecimiento opcional, no como una carencia de la validación realizada.

## Instrumento reproducible

Plantilla base:

```powershell
.venv-nmt/Scripts/python -m scripts.nmt.75_human_validation_workbook
```

Archivo generado originalmente:

| Campo | Valor |
|---|---|
| Plantilla Excel | `reports/05_nmt/evaluation_xl/human_validation_100.xlsx` |
| Ejecución completada | `reports/05_nmt/evaluation_xl/validacion_participativa_100.xlsx` |
| Predicciones fuente | `reports/05_nmt/reranking_xl/nllb_bidi_lora_v2_1b_loraplus_xl/test_predictions_reranked.jsonl` |

## Estructura

El libro contiene una hoja de instrucciones y dos hojas de revisión:

| Hoja | Filas | Dirección |
|---|---:|---|
| `shw2spa` | 50 | shiwilu->castellano |
| `spa2shw` | 50 | castellano->shiwilu |

Columnas visibles: `N`, `Texto fuente`, `Traducción del modelo`, `Sentido (1-5)`, `Naturalidad (1-5)`, `Decisión`, `Comentarios`.

## Rúbrica

| Dimensión | Escala | Criterio |
|---|---|---|
| Sentido | 1-5 | Preservación del significado; penaliza omisiones, agregados y cambios semánticos. |
| Naturalidad | 1-5 | Gramaticalidad, naturalidad y legibilidad en la lengua destino. |
| Decisión | aceptar/corregir/rechazar | Juicio práctico sobre si la salida puede usarse, requiere edición puntual o debe descartarse. |
| Comentarios | texto libre | Correcciones, explicación de errores o dudas lingüísticas. |

## Resultados de la primera ejecución

| Dirección | n | Sentido prom. | Naturalidad prom. | Aceptar | Corregir | Rechazar |
|---|---:|---:|---:|---:|---:|---:|
| shw->spa | 50 | 4.74 | 4.76 | 41 (82%) | 9 (18%) | 0 (0%) |
| spa->shw | 50 | 4.58 | 4.68 | 39 (78%) | 11 (22%) | 0 (0%) |
| Global | 100 | 4.66 | 4.72 | 80 (80%) | 20 (20%) | 0 (0%) |

Validación programática:

- 50 filas por dirección.
- Puntajes 1-5 completos.
- Decisiones normalizables a `aceptar`, `corregir`, `rechazar`.
- Todos los casos corregibles tienen comentario.

## Lectura

La revisión confirma una asimetría coherente con la evaluación automática: shiwilu->castellano es más estable, mientras castellano->shiwilu exige mayor control morfológico, concordancia y selección léxica. No hubo rechazos; los errores observados fueron recuperables mediante correcciones puntuales.
