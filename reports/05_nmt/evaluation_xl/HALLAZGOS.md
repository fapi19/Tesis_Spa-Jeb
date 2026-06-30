# Hallazgos de evaluación del sistema NMT

Documento interno de consolidación para la variante `nllb_bidi_lora_v2_1b_loraplus_xl` sobre `xl`. Resume la evidencia automática, cualitativa y participativa disponible.

## 1. Evaluación automática

El modelo campeón es `v2.1b` LoRA+ con reranking semántico. Su métrica global principal fue:

- avg chrF++ = 44.99
- IC 95% = [43.17, 46.96]
- BLEU promedio = 18.46
- COMET promedio = 0.752

La lectura por dirección sigue siendo necesaria:

- shw->spa: BLEU es más informativo porque la salida está en castellano.
- spa->shw: chrF++ es más fiable porque BLEU penaliza con dureza la morfología aglutinante del shiwilu.

El modelo base `v0` estaba cerca del ruido en BLEU; el campeón supera ese umbral con claridad y produce traducciones informativas.

## 2. Análisis cualitativo

La banda alta confirma que el sistema traduce con contenido, no solo reproduce palabras frecuentes. En shw->spa, una proporción mayor de salidas cae en rangos altos de BLEU. En spa->shw, chrF++ muestra coincidencias parciales útiles incluso cuando BLEU subestima la calidad por cambios morfológicos.

Los errores más visibles son:

- pérdida o cambio de sentido en frases largas;
- errores de persona, número o dirección argumental;
- selección léxica imprecisa;
- formas shiwilu incompletas o poco naturales en spa->shw.

## 3. Validación participativa ejecutada

La primera ejecución completa del protocolo se realizó con Fidel Lomas Chota, hablante/revisor competente y uno de los principales exponentes y preservadores contemporáneos del shiwilu vinculados al proyecto.

Archivo completo:

- `reports/05_nmt/evaluation_xl/validacion_participativa_100.xlsx`
- Enlace externo: https://docs.google.com/spreadsheets/d/1dYJ_4qstg76dHtDs6qaIICG5aBeiG7kL/edit?usp=sharing&ouid=110915706047387486036&rtpof=true&sd=true

Muestra:

- 100 traducciones revisadas.
- 50 shiwilu->castellano.
- 50 castellano->shiwilu.
- Modelo evaluado: `v2.1b` LoRA+ con reranking semántico.

Resultados agregados:

| Dirección | n | Sentido prom. | Naturalidad prom. | Aceptar | Corregir | Rechazar |
|---|---:|---:|---:|---:|---:|---:|
| shw->spa | 50 | 4.74 | 4.76 | 41 (82%) | 9 (18%) | 0 (0%) |
| spa->shw | 50 | 4.58 | 4.68 | 39 (78%) | 11 (22%) | 0 (0%) |
| Global | 100 | 4.66 | 4.72 | 80 (80%) | 20 (20%) | 0 (0%) |

Validación programática del Excel:

- 50 filas por dirección.
- Puntajes 1-5 completos.
- Decisiones normalizables.
- Comentarios presentes en todos los casos corregibles.

## 4. Interpretación

La validación humana confirma la asimetría vista en las métricas automáticas. Traducir desde shiwilu hacia castellano resultó más estable: mayor promedio de sentido, mayor naturalidad y mayor porcentaje de aceptación. Traducir desde castellano hacia shiwilu también fue favorable, pero concentró más correcciones por morfología, concordancia y selección léxica.

No hubo rechazos. Esto es importante: los errores detectados no fueron salidas inutilizables, sino casos recuperables mediante correcciones puntuales.

Para una lengua con muy pocos hablantes disponibles, una revisión completa de 100 ítems por un revisor de la trayectoria de Fidel Lomas Chota aporta evidencia valiosa, trazable y metodológicamente pertinente. El mismo instrumento podría repetirse con más revisores si la disponibilidad de hablantes lo permite, reutilizando la muestra y la rúbrica para calcular acuerdo interevaluador. Esa ampliación se entiende como fortalecimiento opcional, no como una carencia de la primera ejecución.

## 5. Estado de artefactos

| Artefacto | Estado |
|---|---|
| Métricas automáticas por dirección | Ejecutadas |
| Bootstrap CI | Ejecutado |
| Análisis cualitativo estratificado | Ejecutado |
| Instrumento de preferencia pareada | Generado |
| Protocolo de validación humana | Documentado |
| Validación participativa 100 ítems | Ejecutada |
| Ampliación opcional multi-revisor | Posible extensión futura |

## 6. Reproducción

```powershell
# Análisis cualitativo estratificado
.venv-nmt/Scripts/python -m scripts.nmt.42_qualitative_analysis --variant xl

# Instrumento de preferencia pareada ciega
.venv-nmt/Scripts/python -m scripts.nmt.76_pairwise_preference --variant xl

# Plantilla de validación humana
.venv-nmt/Scripts/python -m scripts.nmt.75_human_validation_workbook
```
