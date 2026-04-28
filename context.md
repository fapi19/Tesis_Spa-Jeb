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
