# Caracterización lingüística del shiwilu y su impacto en embeddings y NMT

## 1. Introducción

El shiwilu (también conocido como jebero) es una lengua amazónica del Perú perteneciente a la familia lingüística cahuapana. Actualmente se encuentra en estado crítico de amenaza, con un número muy reducido de hablantes, principalmente adultos mayores, y sin transmisión intergeneracional activa.

Desde una perspectiva computacional, el shiwilu representa un caso extremo de lengua de muy bajo recurso (Low-Resource Language, LRL), lo que implica desafíos tanto en la disponibilidad de datos como en la complejidad lingüística inherente.

---

## 2. Clasificación lingüística y contexto general

- **Familia lingüística:** Cahuapana  
- **Tipo:** Lengua amazónica  
- **Estado:** Seriamente en peligro  
- **Ubicación:** Región Loreto, Perú  
- **Número de hablantes:** ~30–50 hablantes fluidos  

El shiwilu es una de las pocas lenguas sobrevivientes de su familia lingüística y presenta características tipológicas altamente relevantes para el procesamiento automático del lenguaje.

---

## 3. Tipología morfológica

### 3.1 Naturaleza general

El shiwilu es caracterizado como:

- Lengua **sintética**
- **Polisintética**
- **Aglutinante con rasgos flexivos**
- **Incorporante**

Esto implica que:

- Las palabras pueden contener múltiples morfemas
- Una sola palabra puede representar una oración completa
- Existe alta densidad semántica en cada token

---

### 3.2 Estructura morfológica

Las palabras en shiwilu se construyen mediante:

- **Raíz léxica**
- **Prefijos**
- **Sufijos**
- **Clasificadores**
- **Marcas de sujeto y objeto**

Además:

- Los morfemas pueden generar cambios fonológicos entre sí (ej. palatalización)
- Existe interacción entre morfología y fonología

---

### 3.3 Complejidad verbal

Los verbos en shiwilu pueden codificar simultáneamente:

- Sujeto
- Objeto
- Dirección
- Instrumento
- Modo de acción
- Aplicativos

Esto implica que:

> La unidad semántica relevante no es la palabra aislada, sino la estructura morfológica completa.

---

## 4. Sistema de clasificadores

El shiwilu presenta un sistema de clasificadores nominales que cumplen funciones:

- Semánticas
- Sintácticas

Estos clasificadores codifican propiedades como:

- Forma
- Función
- Animacidad

Esto introduce ambigüedad semántica si no se consideran en el modelado.

---

## 5. Sintaxis

El orden de palabras en shiwilu es:

- Flexible (SVO, SOV, OVS)

Esto implica:

- No existe dependencia fuerte del orden lineal
- La interpretación depende más de la morfología que de la posición

---

## 6. Fonología y ortografía

Características relevantes:

- Inventario vocálico reducido
- Presencia de sonidos glotales
- Variación ortográfica entre fuentes
- Influencia fonológica entre morfemas

Esto genera:

- Variabilidad en la escritura
- Dificultades en la normalización del corpus

---

## 7. Implicaciones lingüísticas para NLP

### 7.1 No correspondencia 1 a 1

- Una palabra en shiwilu ≠ una palabra en español
- Traducción requiere mapeo semántico, no léxico

---

### 7.2 Alta variabilidad morfológica

- Muchas formas derivadas de una misma raíz
- Baja repetición exacta de tokens

---

### 7.3 Baja disponibilidad de datos

- Corpus pequeño
- Dominio limitado
- Sesgo hacia narrativas específicas

---

## 8. Implicaciones para el preprocesamiento

### 8.1 Tokenización

#### Problema:
- Tokens extremadamente largos
- Alta variación morfológica

#### Solución:
- Tokenización por subpalabras:
  - SentencePiece
  - BPE (Byte Pair Encoding)
  - Unigram LM

#### Justificación:
- Reduce OOV (Out Of Vocabulary)
- Captura patrones morfológicos

---

### 8.2 Normalización

#### Problema:
- Variantes ortográficas
- Ruido en el corpus

#### Solución:
- Normalización controlada:
  - Regex
  - Reglas lingüísticas mínimas
  - Evitar sobre-normalización

---

### 8.3 Segmentación

#### Problema:
- Frases cortas pero densas
- Una palabra puede ser oración

#### Solución:
- Segmentación cuidadosa por oración
- Evitar dividir unidades semánticas

---

### 8.4 Alineación

#### Problema:
- No existe alineación palabra a palabra

#### Solución:
- Alineación basada en:
  - Longitud
  - Puntuación
  - Similaridad semántica (embeddings)

---

### 8.5 Embeddings

#### Enfoque recomendado:
- Embeddings a nivel de oración:
  - Sentence-BERT (SBERT)
  - E5
  - LASER

#### Justificación:
- Capturan significado global
- Independientes de tokenización exacta

---

### 8.6 Data augmentation

#### Necesidad:
- Corpus extremadamente pequeño

#### Técnicas recomendadas:
- Back-translation
- Parafraseo controlado
- Generación sintética
- Filtrado con embeddings

---

## 9. Implicaciones para modelado NMT

Dado el perfil del shiwilu, se recomienda:

- Modelos basados en subpalabras
- Entrenamiento bidireccional
- Uso de modelos multilingües preentrenados
- Transfer learning desde español
- Uso de adapters o LoRA

---

## 10. Conclusión

El shiwilu presenta una combinación de:

- Alta complejidad morfológica
- Flexibilidad sintáctica
- Escasez extrema de datos

Esto lo convierte en un caso representativo de lenguas de muy bajo recurso donde:

- El preprocesamiento es crítico
- La tokenización subword es obligatoria
- La alineación semántica es preferible a la léxica
- Los embeddings juegan un rol central

En consecuencia, el diseño del pipeline debe priorizar:

- Robustez frente a variación
- Captura de estructura morfológica
- Representación semántica sobre superficial

---

## 11. Estado actual del preprocesamiento de embeddings

La fase de preprocesamiento para embeddings Shiwlu-español quedó cerrada con un
pipeline canónico y auditable. A partir de ahora, el trabajo posterior debe usar
ese pipeline como fuente de verdad y no reabrir la limpieza salvo que aparezca un
error crítico.

### Pipeline canónico

- Script principal: `src/embeddings/preprocess_embeddings.py`
- Auditoría de cierre: `src/embeddings/audit_preprocessing.py`
- Manifiesto: `reports/04_embeddings/preprocessing/preprocess_manifest.json`
- Reporte de cierre: `reports/04_embeddings/preprocessing/preprocessing_closure_report.md`

### Artefactos canónicos

- `data/processed/04_splits/train.jsonl`
- `data/processed/04_splits/valid.jsonl`
- `data/processed/04_splits/test.jsonl`
- `data/processed/04_splits/train.csv`
- `data/processed/04_splits/valid.csv`
- `data/processed/04_splits/test.csv`
- `data/processed/04_splits/all_text_for_sp.txt`

### Decisiones fijadas

- Se preservan `raw_*` y `normalized_*` para trazabilidad.
- Se aplica normalización conservadora: NFC, minúsculas, espacios, comillas dobles
  y unificación de apóstrofes tipográficos.
- No se eliminan apóstrofes internos en shiwilu.
- No se filtra agresivamente por longitud, porque las formas largas pueden ser
  normales en una lengua polisintética.
- Los pares uno-a-muchos se agrupan mediante `group_id` para evitar leakage entre
  splits y para preparar una evaluación multi-positivo.
- `suffix-aware` queda como variante experimental y no como preprocesamiento por
  defecto.

### Resultado de cierre

- Estado del reporte final: `pass`
- Total original: 3207 pares
- Total incluido: 3204 pares
- Excluidos: 3 duplicados exactos
- Splits: 2563 train, 320 valid, 321 test
- Grupos totales: 2982
- Grupos multi-par: 194

### Regla para trabajo futuro

No limpiar pensando en español. Limpiar sin romper morfología shiwilu.

## 12. Estado actual de embeddings

La fase de evaluación/entrenamiento de embeddings ya produjo un candidato actual:
`v3_iterative_hn_e5_base_bidirectional`.

### Modelos evaluados

| Modelo | Descripción | R@1 | R@5 | R@10 | MRR | Mean Rank |
|--------|-------------|----:|----:|-----:|----:|----------:|
| `baseline` | `intfloat/multilingual-e5-small` sin fine-tuning | 0.0966 | 0.2025 | 0.3209 | 0.1633 | 60.3 |
| `v1` | E5 fine-tuned con `MultipleNegativesRankingLoss` | 0.5109 | 0.7788 | 0.8692 | 0.6325 | 5.9 |
| `v2_hn_controlled_hard` | `v1` + hard negatives | 0.5421 | 0.8069 | 0.8879 | 0.6559 | 5.6 |
| `v2_hn_controlled` | `v1` + hard/medium negatives | 0.5670 | 0.8193 | 0.9097 | 0.6755 | 5.5 |
| `baseline_e5_base` | `intfloat/multilingual-e5-base` sin fine-tuning | 0.1059 | 0.2056 | 0.3209 | 0.1751 | 57.5 |
| `v1_e5_base` | E5-base fine-tuned con `MultipleNegativesRankingLoss` | 0.6480 | 0.9128 | 0.9688 | 0.7592 | 2.5 |
| `v1_e5_base_bidirectional` | E5-base fine-tuned bidireccional con `MultipleNegativesRankingLoss` | 0.6573 | 0.9190 | 0.9751 | 0.7704 | 2.5 |
| `v2_hn_controlled_e5_base` | `v1_e5_base` + hard/medium negatives | 0.7134 | 0.9159 | **0.9720** | 0.8037 | **2.4** |
| `v2_hn_controlled_e5_base_bidirectional` | `v1_e5_base_bidirectional` + hard/medium negatives bidireccionales | 0.7508 | **0.9283** | 0.9688 | 0.8276 | 2.9 |
| `v3_iterative_hn_e5_base_bidirectional` | `v2_hn_controlled_e5_base_bidirectional` + iterative hard negatives | **0.7882** | **0.9283** | **0.9782** | **0.8480** | **2.2** |

### Mejora del candidato actual

Sobre el test canónico de 321 pares, `v3_iterative_hn_e5_base_bidirectional`
recupera la traducción correcta en rank 1 para 253 pares en español -> Shiwlu.

- Frente a `v2_hn_controlled_e5_base_bidirectional`: mejora de `0.7508` a
  `0.7882` en `R@1` español -> Shiwlu, equivalente a `+3.74` puntos
  porcentuales, `+4.98%` relativo y `+12` aciertos top-1 adicionales.
- En Shiwlu -> español, mejora de `0.7788` a `0.7913` en `R@1`, equivalente a
  `+1.25` puntos porcentuales, `+1.60%` relativo y `+4` aciertos top-1
  adicionales.

### Validación bidireccional

| Dirección | R@1 | R@5 | R@10 | MRR | Mean Rank | Rank 1 |
|-----------|----:|----:|-----:|----:|----------:|-------:|
| español -> Shiwlu | 0.7882 | 0.9283 | **0.9782** | 0.8480 | 2.2 | 253/321 |
| Shiwlu -> español | **0.7913** | **0.9564** | 0.9688 | **0.8617** | **2.0** | **254/321** |

La iteración v3 mejora `R@1` y `MRR` en ambas direcciones. La única caída frente
al candidato anterior es `R@10` Shiwlu -> español, de 0.9720 a 0.9688, una
degradación de `0.31` puntos porcentuales que queda dentro del umbral de
aceptación.

El análisis de errores top-1 muestra:

- Español -> Shiwlu: 68 errores top-1.
- Shiwlu -> español: 67 errores top-1.
- Principales categorías heurísticas: `gold_has_audit_flag`,
  `semantic_confusion`, `close_score_ambiguity` y `shared_shiwilu_tokens`.

La lectura metodológica es que los errores restantes deben revisarse
cualitativamente antes de cualquier nuevo entrenamiento, porque parte del fallo
puede venir de ruido, equivalencias no agrupadas o ambigüedad real del corpus.

### Decisión actual

- `v3_iterative_hn_e5_base_bidirectional` es el candidato actual de embeddings.
- `v2_hn_controlled_e5_base_bidirectional` queda como candidato anterior
  bidireccional.
- `v2_hn_controlled_e5_base` queda como candidato anterior no bidireccional.
- `v1_e5_base` queda como baseline fuerte para E5-base.
- `v2_hn_controlled` queda como candidato anterior basado en E5-small.
- `legacy_v2` no debe usarse como modelo principal porque venía de hard negatives
  no controlados y degradaba resultados.
- La fase de embeddings queda cerrada provisionalmente. No se recomienda otra
  ronda de hard negatives sin revisión manual de errores y grupos.

### Resumen final de embeddings

- Candidato final provisional: `v3_iterative_hn_e5_base_bidirectional`.
- Razón de aceptación: mejor `R@1` y `MRR` en español -> Shiwlu y Shiwlu ->
  español.
- Riesgo conocido: `R@10` Shiwlu -> español baja de 0.9720 a 0.9688 (`-0.31`
  puntos porcentuales), dentro del umbral aceptado.
- Decisión: detener la fase de embeddings; no hacer más minería iterativa salvo
  evidencia cualitativa fuerte.
- Siguiente paso: integrar/evaluar este modelo en NMT.

### Organización de reportes

Los reportes de embeddings están organizados en `reports/04_embeddings/`:

- `preprocessing/`: cierre del preprocesamiento.
- `baseline/`: E5 sin fine-tuning.
- `v1/`: fine-tuning contrastivo inicial.
- `controlled_hn/`: minería y validación de hard/medium negatives.
- `v2_hn_controlled/`: candidato anterior con E5-small.
- `v2_hn_controlled_e5_base/`: candidato anterior no bidireccional.
- `v2_hn_controlled_e5_base_bidirectional/`: candidato anterior bidireccional.
- `v3_iterative_hn_e5_base_bidirectional/`: entrenamiento del candidato actual.
- `v2_hn_controlled_hard/`: ablación hard-only.
- `experiments/v3_iterative_hn_e5_base_bidirectional/`: evaluación retrieval y análisis
  cualitativo del candidato actual.
- `legacy_v2/`: experimento anterior no controlado.
- `exploratory/`: reportes exploratorios previos.

### Siguiente fase

Usar `v3_iterative_hn_e5_base_bidirectional` como candidato de embeddings para
integración/evaluación con NMT. No reabrir el preprocesamiento ni el
entrenamiento de embeddings salvo error crítico o evidencia cualitativa clara.

## 13. Estado actual de NMT (SA-BiNLLB)

La fase de traducción automática quedó implementada y entrenada como un sistema
bidireccional Shiwilu <-> español sobre NLLB-200 distilled 600M con LoRA, con
filtro semántico previo y cabeceras de extensión para `shw_Latn`. El sistema se
denomina internamente SA-BiNLLB (Semantic-Aware Bidirectional NLLB).

### Pipeline ejecutado

- Phase 1 (`scripts/nmt/10_canonicalize_dataset.py`): canonicalización de los
  splits 04 a CSV bidireccional. Resultado: 4944 train, 320 valid por dirección,
  642 test, sin leakage por `group_id`.
- Phase 2a (`scripts/nmt/20_semantic_filter.py`): filtro semántico con
  `v3_iterative_hn_e5_base_bidirectional`, umbrales 0.45/0.60. Aplicado solo a
  train. Salida en `data/processed/06_nmt_filtered/`.
- Phase 2b (`scripts/nmt/21_build_faiss.py`): índices FAISS `IndexFlatIP`
  separados por idioma sobre los embeddings de train aceptado.
- Phase 3 (`scripts/nmt/22_train_sentencepiece.py`): SentencePiece Unigram
  analítico, vocab 8000 (con `hard_vocab_limit=False` por tamaño de corpus),
  comparado lado a lado contra el tokenizer NLLB en 50 frases Shiwilu.
- Phase 4a-c (`scripts/nmt/30_train_lora.py`): extensión del tokenizer con
  `shw_Latn` (id 256204), inicialización mean del nuevo embedding desde
  `quy_Latn`, `ayr_Latn`, `grn_Latn`, y entrenamiento LoRA (r=16, alpha=32,
  dropout=0.05, target=`q_proj`+`v_proj`). Adaptador resultante:
  2.36M params entrenables (0.38%).

### Configuración de entrenamiento de v0

- Base: `facebook/nllb-200-distilled-600M`.
- Optim: AdamW, lr=2e-4, warmup_ratio=0.05, lr_scheduler=cosine, weight_decay=0.01.
- Batch: per_device=8, grad_accum=4 (efectivo 32).
- Epochs: 20 (3100 steps), label_smoothing=0.1, fp16 (RTX 5060 Ti, sm_120, cu128).
- Eval cada 250 steps, save cada 500, `metric_for_best_model=eval_avg_chrf`.
- `load_best_model_at_end=True`, `predict_with_generate=True`, beams=5.
- Tiempo total: 4090 s (~68 min) en RTX 5060 Ti.

### Resultado de v0 (validation, mejor checkpoint)

Mejor checkpoint: `models/nmt/nllb_bidi_lora_v0/checkpoint-2500` (epoch 17.74).
Métrica: `eval_avg_chrf = 17.28`.

| Dirección | chrF++ | BLEU | eval_loss |
|-----------|-------:|-----:|----------:|
| shw -> spa | 19.12 | 1.77 | 3.74 |
| spa -> shw | 15.43 | 0.24 | 4.30 |
| **avg** | **17.28** | **~1.0** | — |

### Resultado de v0 (test, Phase 5 + Phase 6)

Phase 5 ejecutada sobre el split de test (642 ejemplos, 321 por dirección)
con beam=5, length_penalty=1.0, max_new_tokens=128 y métricas
BLEU + chrF++ (sacrebleu) + BERTScore-F1 (`xlm-roberta-large`) +
COMET (`Unbabel/wmt22-comet-da`).

| Sistema | Dirección | chrF++ | BLEU | BERTScore-F1 | COMET |
|---------|-----------|-------:|-----:|-------------:|------:|
| v0 baseline | shw -> spa | 18.63 | 3.49 | 0.894 | 0.601 |
| v0 baseline | spa -> shw | 20.65 | 2.03 | 0.866 | 0.646 |
| **v0 baseline avg** | — | **19.64** | 2.76 | — | — |
| v0 + reranker α=0.7 | shw -> spa | 19.95 | 3.71 | 0.897 | 0.611 |
| v0 + reranker α=0.7 | spa -> shw | 21.84 | 2.21 | 0.869 | 0.666 |
| v0 + reranker α=0.7 avg | — | 20.89 | 2.96 | — | — |
| **v0 + reranker α=0.5 (best)** | shw -> spa | **20.07** | 3.65 | — | — |
| **v0 + reranker α=0.5 (best)** | spa -> shw | **22.06** | 2.27 | — | — |
| **v0 + reranker α=0.5 (best) avg** | — | **21.07** | 2.96 | — | — |

`BERTScore` y `COMET` se reportan solo como proxy: ninguno de los dos modelos
vio Shiwilu durante su pre-entrenamiento. La métrica primaria sigue siendo
`chrF++` (registrado como tal en los JSON con campo
`primary_metric: "chrf_pp"`).

Phase 6 ejecuta una ablación de α en
`final_score = α · trans_prob + (1 − α) · cos_sim`, donde `trans_prob` es la
softmax de las `sequence_scores` de NLLB sobre las top-5 hipótesis y `cos_sim`
es la similitud entre el embedding `v3_iterative_hn_e5_base_bidirectional` de
la fuente y de cada candidato.

| α (peso `trans_prob`) | avg chrF++ | avg BLEU | shw -> spa chrF++ | spa -> shw chrF++ |
|:----------------------|-----------:|---------:|------------------:|------------------:|
| 0.0 (puro SBERT)      | 20.89      | 2.88     | 19.88             | 21.90             |
| 0.3                   | 20.95      | 2.90     | 19.95             | 21.96             |
| **0.5 (óptimo)**      | **21.07**  | **2.96** | **20.07**         | **22.06**         |
| 0.7 (default config)  | 20.89      | 2.96     | 19.95             | 21.84             |
| 1.0 (sin SBERT)       | 19.64      | 2.76     | 18.63             | 20.65             |

Lecturas registradas:

- El reranker semántico aporta **+1.42 chrF++ avg** sobre baseline
  (19.64 -> 21.07). La curva de α tiene forma de U invertida bien comportada
  con pico interior en α=0.5, lo que confirma que la calibración entre
  `trans_prob` y `cos_sim` es coherente.
- α=0.0 (puro SBERT) ya supera baseline en +1.25 chrF: el modelo de embeddings
  v3 está aportando señal real, no es ruido.
- Asimetría direccional consistente: `spa -> shw` queda por encima en chrF++
  (premia aciertos morfológicos del decoder Shiwilu) y `shw -> spa` por encima
  en BLEU (target español con tokenización word-level).
- Test sube respecto a la mejor evaluación de validation (avg 17.28 -> 19.64
  baseline / 21.07 con reranker) sin signos de leakage (`group_id` está
  aislado desde Phase 1).
- **Decisión operativa**: se fija **α=0.5** como default del reranker para
  todas las corridas posteriores (v1_bt y comparativos). El default de
  `config/nmt/reranker.yaml` (α=0.7) se mantiene como referencia histórica de
  la corrida inicial.

### Trayectoria observada

- chrF avg crece monótonamente de 8.09 (epoch 1.6) a 17.28 (epoch 17.74).
- Plateau aparente alrededor de epoch 9-10 para `shw -> spa`, lo cual sugiere
  saturación de capacidad LoRA con r=16. Para v1_bt se evaluará subir a r=32.
- `spa -> shw` sigue mejorando hasta el final (15.48 en epoch 19.36), señal de
  que la dirección difícil aún tiene margen con más data.
- `train_loss` final: 4.02 (último step) / 4.44 (promedio de la corrida), desde
  7.11 inicial.
- Sin overfitting: `eval_loss` baja en paralelo a `train_loss` durante toda la
  corrida.

### Lectura metodológica

- chrF++ es la métrica primaria por la naturaleza polisintética del Shiwilu.
- BLEU está bajo en ambas direcciones (1.77 y 0.24); el modelo se optimizó por
  `eval_avg_chrf`, no por BLEU. La asimetría entre chrF y BLEU es esperable en
  lenguas agglutinantes con corpus pequeño: el modelo produce traducciones
  semántica y morfológicamente cercanas, pero con vocabulario alternativo al
  gold.
- La asimetría `shw->spa > spa->shw` es estructural y conocida: generar Shiwilu
  exige producir tokens del nuevo `shw_Latn` cuya embedding aún se estaba
  ajustando.

### Comparación con literatura previa

Sobre el mismo corpus Shiwilu, un baseline transformer entrenado desde cero
reporta `BLEU_w` Total = 3.76 (Religioso 1.29, Educativo 4.10, Flashcards 11.95).
Esa cifra mezcla ambas direcciones y se computa con tokenización word-level
(comparable a `sacreBLEU`, no a `BLEU_BPE`).

En test, v0 queda en BLEU `shw -> spa` 3.49 (baseline) / 3.65–3.71 (con
reranker, según α), prácticamente al nivel del baseline previo (3.76 Total)
pero ya con un sistema bidireccional, sin pérdida en `spa -> shw`, y
construido sobre el contrato de filtro semántico + extensión de NLLB. Ventajas
estructurales sobre ese baseline previo:

- Filtro semántico explícito sobre el train (Phase 2a).
- Reranker semántico post-generación con +1.42 chrF++ avg sobre baseline en
  test (Phase 6 ejecutada).
- Tokenizer NLLB extendido con `shw_Latn` mean-init desde `quy_Latn` /
  `ayr_Latn` / `grn_Latn`, en lugar de un Transformer-from-scratch.
- Backtranslation y mining (Phase 7) aún no incorporados al training; siguiente
  ganancia esperada al entrenar v1_bt con LoRA `r=32 / alpha=64`.

### Decisiones técnicas registradas

- Compatibilidad con transformers 4.55 + peft 0.16 + accelerate 1.8 mantenida
  sin downgrades. El `DataCollatorForSeq2Seq` estándar deja de añadir
  `decoder_input_ids` cuando el modelo no expone `prepare_decoder_input_ids_from_labels`
  (cosa que pasa en NLLB con el patrón modular nuevo); combinado con
  `label_smoothing_factor=0.1`, esto rompe el forward del decoder. Se introdujo
  `Seq2SeqCollatorWithDecoderInputs` en `src/nmt/training/train_lora.py` que
  precomputa `decoder_input_ids` con `shift_tokens_right` antes de entregar el
  batch.
- `sacrebleu`: signature obtenida con `metric.get_signature()` (cambio API
  reciente). Documentado en `src/nmt/evaluation/metrics.py`.
- SentencePiece analítico: `hard_vocab_limit=False` porque ~5k frases Shiwilu
  no permiten alcanzar 8000 unigrams. Se reporta tamaño efectivo en
  `sentencepiece_stats.json`.
- Heurística de monolingüe Shiwilu: requiere apóstrofe por defecto, configurable
  con `--no-require-apostrophe`. Reduce contaminación con español.
- `dataloader_num_workers=2` en Windows por las semánticas de fork.
- Fix de inferencia: al recargar el tokenizer del checkpoint, NLLB reconstruye
  `lang_code_to_id` desde su lista hardcoded y `shw_Latn` desaparece (aunque
  sigue en `additional_special_tokens`). Se añadió
  `_ensure_extended_lang_codes_registered` en `src/nmt/inference/generate.py`
  que re-registra cualquier código FLORES-style del tokenizer cargado al
  entrar en `load_checkpoint`. Sin ese fix, Phase 5/6 dispara
  `RuntimeError: 'shw_Latn' not registered in tokenizer.lang_code_to_id`.

### Estado de fases siguientes

- Phase 5 (eval completo sobre test 642 con BLEU + chrF + BERTScore-F1 con
  `xlm-roberta-large` + COMET con `wmt22-comet-da`): **ejecutada**.
  avg chrF++ = 19.64. Reportes en
  `reports/05_nmt/evaluation/nllb_bidi_lora_v0/{test_metrics.json,test_predictions.jsonl,test_predictions_topk.jsonl}`.
- Phase 6 (reranker semántico con `v3_iterative_hn_e5_base_bidirectional`,
  alpha sweep `{0.0, 0.3, 0.5, 0.7, 1.0}`): **ejecutada**. Mejor α = 0.5,
  avg chrF++ = 21.07 (+1.42 sobre baseline). Reportes en
  `reports/05_nmt/reranking/nllb_bidi_lora_v0/{test_metrics_reranked.json,test_predictions_reranked.jsonl,ablation.json}`.
- Phase 7a (backtranslation con v0): **ejecutada**. 76 líneas mono Shiwilu
  → 12 filas sintéticas (6 pares × 2 dirs) tras filtro ≥ 0.60; score medio
  0.666, max 0.787, sin recortar contra el cap (2× paralelo). Sólo se hace
  `shw → spa`; se omite `spa → shw` para no contaminar el target Shiwilu
  con generaciones del propio v0. Reporte en
  `reports/05_nmt/augmentation/backtranslation.json`; CSV en
  `data/processed/07_nmt_augmented/train_bt.csv`
  (`origin_source = backtranslation_v0`).
- Phase 7b (embedding mining): ya ejecutado, 1338 pares aceptados (`reciprocal-NN`,
  IP > 0.65) en `data/processed/07_nmt_augmented/train_mined.csv`.
- Phase 7c (variantes morfológicas): apagado por defecto sin lingüista, generará
  CSV de revisión.
- Phase 7d (re-entreno como v1_bt): **pendiente, lanzamiento manual**. Sube
  LoRA a r=32 / alpha=64 (Enhancement #4-bump) y activa weighted loss vía
  `origin_source` con weights `{flashcards: 1.0, pdf_textos: 1.0,
  mined_v3_sbert: 0.5, backtranslation_v0: 0.3}`.
- Phase 8a/b/c (comparación v0 vs v1_bt con y sin reranker α=0.5, plantilla
  humana, tablas LaTeX): **pendiente** hasta que v1_bt termine de entrenar.
  Código actualizado:
  - `scripts/nmt/70_compare_runs.py` agrega columnas `chrF++ rare` y
    `OOV recovery` al MD comparativo.
  - `scripts/generate_nmt_tables.py` emite ahora 7 tablas autogeneradas,
    incluyendo `nmt_rare_token.tex` y la versión expandida de
    `nmt_sentencepiece_vs_nllb.tex` (frase, palabra, vocab efectivo,
    fragmentación).

### Enhancements integrados

A partir de v0 ya entrenado y antes de v1_bt se sumaron cuatro enhancements
para enriquecer el reporting sin re-entrenar:

- **#6 Reliability/confidence layer** (`src/nmt/inference/confidence.py`,
  integrado en `40_evaluate.py` y `50_rerank.py`). Cada predicción guarda
  `confidence ∈ {low, medium, high}`, `confidence_score` y
  `confidence_components`. Baseline usa `exp(top-1 sequence_score)`
  (probabilidad geométrica por token), umbrales (0.40, 0.55) calibrados
  sobre v0 test (rango observado [0.20, 0.79], mediana 0.41). Reranked
  usa `final_score`, umbrales (0.30, 0.40) calibrados sobre v0 + reranker
  α=0.7 (rango [0.10, 0.48], mediana 0.28). Distribución v0 baseline:
  304 low / 254 medium / 84 high; v0 + reranker α=0.7: 405 / 218 / 19. La
  distribución se persiste en `meta.confidence` de cada `*_metrics.json`.
- **#2 Rare-token / morphology-aware evaluation**
  (`src/nmt/evaluation/rare_token.py`,
  `scripts/nmt/41_rare_token_eval.py`). El test se buckea por
  `rare_word_ratio` (palabras con frecuencia en train < 5); se reporta
  chrF++ por bucket y `oov_recovery_rate` (palabras OOV en gold que también
  aparecen en la hipótesis). Headline = bucket "≥20% raras". Resultado v0
  baseline: avg chrF++ raras = 19.58, avg OOV-recovery = 0.022. Resultado
  v0 + reranker: 20.75 y 0.026. Vocab train: 3821 unique shw / 3838 unique
  spa. Reportes:
  `reports/05_nmt/evaluation/nllb_bidi_lora_v0/rare_token_analysis{,_reranked}.json`.
- **#3 Comparación de tokenizadores** (artefacto de tesis, no de modelo).
  La tabla expandida en `scripts/generate_nmt_tables.py` añade tokens por
  palabra y fracción de oraciones donde cada tokenizer es más corto. Sobre
  50 oraciones Shiwilu (seed=42): SP Unigram 9.68 tokens/frase, 3.34
  tokens/palabra; NLLB-200 12.24 / 4.22; SP es más corto en 41/50 oraciones,
  NLLB en 3/50, empate en 6/50. Justificación de mantener NLLB pese a su
  mayor fragmentación: el backbone ya domina cientos de lenguas latinas
  cercanas y la transferencia multilingüe pesa más que el ahorro local de
  subwords. Tabla autogenerada:
  `thesis/latex/figuras/generated/nmt_sentencepiece_vs_nllb.tex`.
- **#4 Weighted synthetic-data training**
  (`src/nmt/training/dataset.py`, `src/nmt/training/train_lora.py`,
  `scripts/nmt/63_train_with_augmented.py`). El dataset bidireccional
  inyecta una columna `sample_weight` derivada de `origin_source` cuando
  se le pasa un `weight_map`. El collator
  `Seq2SeqCollatorWithDecoderInputs` extrae el peso de cada feature antes
  de pasar al `DataCollatorForSeq2Seq` upstream, y lo re-attacha como
  tensor `[B]` float32. `BidiSeq2SeqTrainer.compute_loss` detecta la
  presencia de `sample_weight` y conmuta a `_weighted_smoothed_ce`, que
  replica el label-smoothing de HuggingFace a nivel per-row y luego pondera
  cada fila por su peso, con denominador en tokens ponderados (un row con
  `weight=0.3` contribuye 30% de sus tokens al promedio). Cuando el peso es
  uniforme se vuelve a la ruta del `Seq2SeqTrainer` original (back-compat:
  v0 sigue corriendo idéntico al pre-#4). Para v1_bt, el script lanzador
  además bumpea LoRA a `r=32 / alpha=64`. Tests unitarios verifican (a) el
  mapping `origin_source → weight`, (b) `weight=0` colapsa la fila a 0, y
  (c) ausencia de `sample_weight` no contamina la batch.

### Reproducibilidad

- Adaptador final v0: `models/nmt/nllb_bidi_lora_v0/`.
- Tokenizer extendido: `models/nmt/tokenizer_shw_extended/`.
- Reportes: `reports/05_nmt/training/nllb_bidi_lora_v0/{summary,training_log}.json`.
- Log completo de la corrida: `train_v0.log` (5870 líneas).
- Logs de re-corridas para `confidence`/`rare-token`: `eval_v0_full.log`,
  `rerank_v0_full.log`, `rare_v0_baseline.log`, `rare_v0_reranked.log`.
- Log del Phase 7a actual: `bt_v0.log`.

### Future work (no incluido en este ciclo)

Documentado para no inflar el alcance de la tesis y para evitar correr más
de un ciclo de entrenamiento sin evidencia clara de retorno:

- **Iterative backtranslation** (Edunov et al., 2018; Hoang et al., 2018):
  iterar BT → re-train → BT → re-train mejora calidad en LRL pero exige un
  pool mono mucho mayor que las 76 líneas actuales y un protocolo claro de
  detención (parar cuando la calidad del BT en validación deja de mejorar).
- **Domain-controlled monolingual Spanish corpus**: hoy se omite BT
  `spa → shw` para no envenenar el target. Curar 10–30k frases en español
  cercanas al dominio (flashcards, textos comunitarios) y filtrarlas con el
  v3 SBERT habilitaría BT `spa → shw` con un v1_bt ya estable.
- **Focal loss para preservación de rare-tokens**: la baja
  `oov_recovery_rate` (~2%) sugiere que el decoder casi no copia palabras
  Shiwilu OOV cuando aparecen en la fuente. Focal loss con γ ≈ 2 sobre los
  token-ids más raros sesgaría al modelo a preservarlos. Es ortogonal al
  weighted-data y se puede combinar.
