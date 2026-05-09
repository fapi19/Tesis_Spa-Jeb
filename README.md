# NMT Español-Shiwilu (spa-jeb)

Sistema de Traducción Automática Neuronal (Neural Machine Translation) entre
español (`spa`) y shiwilu/jebero (`jeb`), una lengua originaria de la familia
cahuapana hablada en la Amazonía peruana.

El shiwilu es una lengua de bajos recursos con escasos datos digitalizados.
Este proyecto aborda el desafío de construir un sistema de traducción automática
a partir de un corpus paralelo limitado, aplicando técnicas de preprocesamiento,
generación de embeddings bilingües y entrenamiento de modelos neuronales.

---

## Estado actual

La fase de preprocesamiento para embeddings Shiwlu-español está cerrada. El
pipeline canónico quedó en `src/embeddings/preprocess_embeddings.py` y fue
validado con `src/embeddings/audit_preprocessing.py`.

Resumen actual de datos:

- Dataset original: `3207` pares.
- Dataset incluido: `3204` pares.
- Excluidos: `3` duplicados exactos.
- Splits canónicos: `2563` train, `320` valid, `321` test.
- Grupos totales: `2982`.
- Grupos multi-par: `194`.
- Estado de cierre: `pass`.
- `suffix-aware` queda como variante experimental, no como default.

Regla central: no limpiar pensando en español; limpiar sin romper morfología
shiwilu.

Modelo candidato actual:

- `v3_iterative_hn_e5_base_bidirectional`
- Base: `intfloat/multilingual-e5-base`
- Etapas: baseline E5-base -> fine-tuning `v1_e5_base_bidirectional` -> hard/medium negative mining controlado bidireccional -> iterative hard negative mining v3
- Español -> Shiwlu: `R@1=0.7882`, `R@5=0.9283`, `R@10=0.9782`, `MRR=0.8480`
- Shiwlu -> español: `R@1=0.7913`, `R@5=0.9564`, `R@10=0.9688`, `MRR=0.8617`
- Mejora frente a `v2_hn_controlled_e5_base_bidirectional`: `+3.74` puntos
  porcentuales en `R@1` español -> Shiwlu (`+4.98%` relativo; 253 vs 241
  aciertos en rank 1) y `+1.25` puntos porcentuales en `R@1` Shiwlu ->
  español (`+1.60%` relativo; 254 vs 250 aciertos en rank 1).

Decisión: cerrar provisionalmente la fase de embeddings y usar
`v3_iterative_hn_e5_base_bidirectional` como candidato para
integración/evaluación con NMT.

---

## Estructura del proyecto

```
Desarrollo/
├── config/
│   ├── normalization_rules.json        # Reglas de normalización configurables
│   └── sources.json                    # Registro de fuentes de datos
├── data/
│   ├── raw/                            # Datos originales sin modificar
│   │   ├── flashcards2.csv
│   │   └── II_TEXTOS_SHIWILU.pdf
│   ├── intermediate/                   # Datos en proceso (por etapa)
│   │   ├── 00_pdf/
│   │   │   └── dataset_extraido_pdf.csv
│   │   ├── 01_filtrado/
│   │   │   └── dataset_filtrado.csv
│   │   ├── 01b_unificado/
│   │   │   └── dataset_unificado.csv
│   │   └── 02_normalizado/
│   │       └── dataset_normalizado.csv
│   └── processed/                      # Datos finales listos para modelos
│       ├── 03_pre_embeddings/
│       │   └── dataset_pre_embeddings.csv
│       └── 04_splits/
│           ├── train.jsonl             # Split canónico de embeddings
│           ├── valid.jsonl
│           ├── test.jsonl
│           ├── train.csv               # Espejo para Sentence Transformers
│           ├── valid.csv
│           ├── test.csv
│           └── all_text_for_sp.txt     # Corpus para SentencePiece
├── scripts/
│   ├── 00_extraer_dataset_pdf.py       # Etapa 0: Extracción desde PDF
│   ├── 01_filtrar_dataset.py           # Etapa 1: Filtrado inicial (flashcards)
│   ├── 01b_unificar_fuentes.py         # Etapa 1b: Unificación de fuentes
│   ├── 02_depurar_dataset.py           # Etapa 2: Normalización no destructiva
│   └── 03_auditar_dataset.py           # Etapa 3: Auditoría y exportación final
├── src/
│   ├── embeddings/
│   │   ├── preprocess_embeddings.py    # Preprocesamiento canónico
│   │   ├── audit_preprocessing.py      # Auditoría/cierre del preprocesamiento
│   │   ├── run_experiment.py           # Experimentos de embeddings
│   │   ├── train_embedding_model.py    # Encoder contrastivo propio
│   │   └── exploratory/                # Experimentos Sentence Transformers
│   └── nmt/                            # Modelos y utilidades NMT
├── models/
│   ├── fasttext/                       # Embeddings FastText (Skip-Gram)
│   │   ├── fasttext.model              # Modelo completo (gensim)
│   │   └── fasttext.vec                # Vectores formato word2vec
│   └── sentence_transformers/          # Embeddings Sentence Transformers
│       ├── embeddings_esp.npy          # Embeddings oraciones español
│       └── embeddings_shi.npy          # Embeddings oraciones shiwilu
├── notebooks/                          # Exploración interactiva (siguiente fase)
├── reports/                            # Reportes organizados por etapa
│   ├── 00_pdf/
│   │   └── summary.json
│   ├── 01_filtrado/
│   │   └── rows_removed.csv
│   ├── 01b_unificado/
│   │   ├── summary.json
│   │   └── cross_duplicates.csv
│   ├── 02_normalizacion/
│   │   ├── normalization_log.csv
│   │   ├── rows_removed.csv
│   │   └── summary.json
│   ├── 03_auditoria/
│   │   ├── problem_rows.csv
│   │   └── summary.json
│   └── 04_embeddings/
│       ├── README.md                   # Organización de reportes de embeddings
│       ├── preprocessing/              # Manifiesto y cierre del preprocesamiento
│       ├── baseline/                   # E5 sin fine-tuning
│       ├── v1/                         # E5 + MultipleNegativesRankingLoss
│       ├── controlled_hn/              # Minería/validación de negativos
│       ├── v2_hn_controlled/           # Candidato anterior E5-small + HN
│       ├── v2_hn_controlled_e5_base/   # Candidato anterior E5-base + HN
│       ├── v2_hn_controlled_e5_base_bidirectional/ # Candidato anterior bidireccional
│       ├── v3_iterative_hn_e5_base_bidirectional/  # Modelo candidato actual
│       ├── v2_hn_controlled_hard/      # Ablación hard-only
│       ├── legacy_v2/                  # Triplets antiguos no controlados
│       └── exploratory/                # Reportes exploratorios previos
├── pyproject.toml
├── poetry.lock
├── .gitignore
└── README.md
```

---

## Requisitos previos

- **Python 3.12 o superior**
  Verificar con `python --version`. Si no lo tienes, descárgalo desde
  https://www.python.org/downloads/
- **pip** (viene incluido con Python)

---

## Instalación del entorno

Abrir una terminal CMD y ejecutar los siguientes comandos en orden.

### 1. Ir a la carpeta del proyecto

```cmd
cd ruta\al\proyecto\Desarrollo
```

### 2. Instalar Poetry (solo la primera vez)

```cmd
pip install poetry
```

### 3. Configurar Poetry para crear el entorno dentro del proyecto (solo la primera vez)

```cmd
poetry config virtualenvs.in-project true
```

Esto hace que el entorno virtual se cree en una carpeta `.venv/` dentro del
proyecto, lo que facilita encontrarlo y eliminarlo si es necesario.

### 4. Instalar las dependencias

```cmd
poetry install
```

Este comando lee `pyproject.toml`, descarga todas las dependencias con sus
versiones exactas (registradas en `poetry.lock`) y crea el entorno virtual.

---

## Activación del entorno

Cada vez que abras una terminal nueva, necesitas activar el entorno.
Hay dos formas de hacerlo:

### Opción A: Usar `poetry run` (sin activar)

Agrega `poetry run` antes de cada comando. No necesitas activar nada.

```cmd
poetry run python scripts/01_filtrar_dataset.py
```

### Opción B: Activar el entorno una vez y ejecutar sin prefijo

```cmd
.venv\Scripts\activate
```

Verás que la terminal muestra `(.venv)` al inicio de la línea. Desde ese
momento puedes ejecutar comandos directamente sin el prefijo `poetry run`:

```cmd
python scripts/01_filtrar_dataset.py
```

Para desactivar el entorno cuando termines:

```cmd
deactivate
```

---

## Pipeline de preprocesamiento

El pipeline tiene 5 etapas (00, 01, 01b, 02, 03). Cada etapa genera salidas en su propia
subcarpeta, manteniendo trazabilidad y orden. Todas las fuentes convergen en la etapa 01b
y pasan por el mismo proceso de normalización y auditoría.

### Orden de ejecución

```cmd
poetry run python scripts/00_extraer_dataset_pdf.py   # Fuente: PDF
poetry run python scripts/01_filtrar_dataset.py       # Fuente: flashcards
poetry run python scripts/01b_unificar_fuentes.py     # Unificar todas las fuentes
poetry run python scripts/02_depurar_dataset.py       # Normalización (sobre unificado)
poetry run python scripts/03_auditar_dataset.py       # Auditoría y dataset final
```

---

## Etapa 00: Extracción desde PDF

**Script:** `scripts/00_extraer_dataset_pdf.py`

Extrae pares bilingües shiwilu-castellano desde un PDF con estructura numerada.
Usa heurísticas conservadoras para separar los idiomas y marca casos ambiguos.

**Entrada:**
- `data/raw/II_TEXTOS_SHIWILU.pdf`

**Salidas:**
- `data/intermediate/00_pdf/dataset_extraido_pdf.csv`
- `reports/00_pdf/summary.json`

---

## Etapa 01: Filtrado inicial (flashcards)

**Script:** `scripts/01_filtrar_dataset.py`

Filtra el CSV original para quedarse solo con filas que tengan valores válidos
en ambas columnas (ESP y SHIWILU). Asigna un `pair_id` único a cada par.

**Entrada:**
- `data/raw/flashcards2.csv`

**Salidas:**
- `data/intermediate/01_filtrado/dataset_filtrado.csv`
- `reports/01_filtrado/rows_removed.csv`

**Criterios de exclusión:**
- Filas con `ESP` o `SHIWILU` vacío, nulo, o placeholder `"--"`

---

## Etapa 01b: Unificación de fuentes

**Script:** `scripts/01b_unificar_fuentes.py`

Combina todas las fuentes de datos configuradas en un único dataset. Esto permite
que el PDF, flashcards y cualquier fuente futura pasen por el mismo proceso de
normalización y auditoría.

**Entrada:**
- `config/sources.json` (configuración de fuentes)
- `data/intermediate/01_filtrado/dataset_filtrado.csv` (flashcards)
- `data/intermediate/00_pdf/dataset_extraido_pdf.csv` (PDF)

**Salidas:**
- `data/intermediate/01b_unificado/dataset_unificado.csv`
- `reports/01b_unificado/summary.json`
- `reports/01b_unificado/cross_duplicates.csv`

**Columnas del dataset unificado:**
- `pair_id` - ID único unificado (U00000, U00001, ...)
- `ESP` - Texto en español
- `SHIWILU` - Texto en shiwilu
- `source` - Fuente de origen (flashcards, pdf_textos, etc.)
- `source_pair_id` - ID original de la fuente

**Para agregar una nueva fuente:**
1. Crear script de extracción si es necesario
2. Agregar entrada en `config/sources.json`
3. Re-ejecutar desde `01b` en adelante

---

## Etapa 02: Normalización no destructiva

**Script:** `scripts/02_depurar_dataset.py`

Aplica normalización configurable sin destruir el texto original. Mantiene
columnas separadas para texto original y normalizado.

**Entrada:**
- `data/intermediate/01b_unificado/dataset_unificado.csv`
- `config/normalization_rules.json`

**Salidas:**
- `data/intermediate/02_normalizado/dataset_normalizado.csv`
- `reports/02_normalizacion/normalization_log.csv`
- `reports/02_normalizacion/rows_removed.csv`
- `reports/02_normalizacion/summary.json`

**Reglas de normalización (configurables en JSON):**

| Regla | Descripción | Estado por defecto |
|-------|-------------|--------------------|
| `unicode_nfc` | Normalización Unicode NFC | Activa |
| `trim` | Eliminar espacios al inicio/final | Activa |
| `collapse_spaces` | Colapsar espacios múltiples | Activa |
| `normalize_comma_space` | Normalizar ` , ` a `, ` | Activa |
| `lowercase` | Convertir a minúsculas | Activa |

Las reglas destructivas (eliminar puntuación, paréntesis) están **desactivadas**
por defecto para preservar información en esta fase de tesis.

---

## Etapa 03: Auditoría y exportación final

**Script:** `scripts/03_auditar_dataset.py`

Detecta problemas estructurales del corpus y genera el dataset final para
embeddings. Las filas problemáticas se marcan pero NO se eliminan automáticamente.

**Entrada:**
- `data/intermediate/02_normalizado/dataset_normalizado.csv`

**Salidas:**
- `reports/03_auditoria/problem_rows.csv`
- `reports/03_auditoria/summary.json`
- `data/processed/03_pre_embeddings/dataset_pre_embeddings.csv`

**Problemas detectados:**

| Tipo | Descripción |
|------|-------------|
| `empty_field` | Campos vacíos en originales o normalizados |
| `exact_duplicate` | Pares duplicados exactos (mismo ESP + SHIWILU normalizado) |
| `one_to_many_esp` | Mismo ESP con múltiples traducciones SHIWILU |
| `many_to_one_shiwilu` | Mismo SHIWILU con múltiples traducciones ESP |
| `length_issue` | Longitudes extremas o desbalance fuerte ESP/SHIWILU |
| `suspicious_content` | Caracteres sospechosos, paréntesis, glosas, solo números |

**Estadísticas incluidas en el reporte:**
- Distribución de longitudes (palabras por oración)
- Vocabulario único por idioma
- Type-Token Ratio (TTR)
- Hapax legomena

---

## Etapa 04: Preprocesamiento canónico de embeddings

**Script principal:** `src/embeddings/preprocess_embeddings.py`

Genera los splits canónicos para embeddings a partir de
`data/processed/03_pre_embeddings/dataset_pre_embeddings.csv`. Esta etapa ya
quedó cerrada y debe considerarse la fuente de verdad para los experimentos
posteriores.

**Ejecución:**

```cmd
poetry run python -m src.embeddings.preprocess_embeddings
poetry run python -m src.embeddings.audit_preprocessing
```

**Salidas principales:**

- `data/processed/04_splits/train.jsonl`
- `data/processed/04_splits/valid.jsonl`
- `data/processed/04_splits/test.jsonl`
- `data/processed/04_splits/train.csv`
- `data/processed/04_splits/valid.csv`
- `data/processed/04_splits/test.csv`
- `data/processed/04_splits/all_text_for_sp.txt`
- `reports/04_embeddings/preprocessing/preprocess_manifest.json`
- `reports/04_embeddings/preprocessing/preprocessing_closure_report.md`

**Decisiones fijadas:**

- Se preservan columnas `raw_*` y `normalized_*`.
- Se mantienen apóstrofes internos en shiwilu.
- No se filtra agresivamente por longitud.
- Los casos uno-a-muchos se agrupan con `group_id` para evitar leakage.
- Los aliases antiguos de embeddings (`train_pairs.jsonl`, `val_pairs.jsonl`,
  `all_text.txt`) ya no se usan.

---

## Etapa 05: Entrenamiento y evaluación de embeddings

Esta etapa ya produjo el candidato actual `v3_iterative_hn_e5_base_bidirectional`.
Todos los modelos se evaluaron con retrieval multi-positivo por `group_id` y,
para los candidatos finales, en ambas direcciones.

| Modelo | Entrenamiento | R@1 | R@5 | R@10 | MRR | Mean Rank |
|--------|---------------|----:|----:|-----:|----:|----------:|
| `baseline` | E5 sin fine-tuning | 0.0966 | 0.2025 | 0.3209 | 0.1633 | 60.3 |
| `v1` | E5 + `MultipleNegativesRankingLoss` | 0.5109 | 0.7788 | 0.8692 | 0.6325 | 5.9 |
| `v2_hn_controlled_hard` | `v1` + hard negatives | 0.5421 | 0.8069 | 0.8879 | 0.6559 | 5.6 |
| `v2_hn_controlled` | `v1` + hard/medium negatives | 0.5670 | 0.8193 | 0.9097 | 0.6755 | 5.5 |
| `baseline_e5_base` | E5-base sin fine-tuning | 0.1059 | 0.2056 | 0.3209 | 0.1751 | 57.5 |
| `v1_e5_base` | E5-base + `MultipleNegativesRankingLoss` | 0.6480 | 0.9128 | 0.9688 | 0.7592 | 2.5 |
| `v1_e5_base_bidirectional` | E5-base + MNRL bidireccional | 0.6573 | 0.9190 | 0.9751 | 0.7704 | 2.5 |
| `v2_hn_controlled_e5_base` | `v1_e5_base` + hard/medium negatives | 0.7134 | 0.9159 | **0.9720** | 0.8037 | **2.4** |
| `v2_hn_controlled_e5_base_bidirectional` | `v1_e5_base_bidirectional` + hard/medium negatives bidireccionales | 0.7508 | **0.9283** | 0.9688 | 0.8276 | 2.9 |
| `v3_iterative_hn_e5_base_bidirectional` | `v2_hn_controlled_e5_base_bidirectional` + iterative hard negatives | **0.7882** | **0.9283** | **0.9782** | **0.8480** | **2.2** |

La mejora principal del candidato actual está en el ordenamiento top-1. Frente a
`v2_hn_controlled_e5_base_bidirectional`, sube de 241 a 253 aciertos rank 1 en
español -> Shiwlu (`+12` aciertos, `+3.74` puntos porcentuales de `R@1`,
`+4.98%` relativo). En Shiwlu -> español sube de 250 a 254 aciertos rank 1
(`+4` aciertos, `+1.25` puntos porcentuales de `R@1`, `+1.60%` relativo).

### Validación bidireccional del candidato

| Dirección | R@1 | R@5 | R@10 | MRR | Mean Rank | Rank 1 |
|-----------|----:|----:|-----:|----:|----------:|-------:|
| español -> Shiwlu | 0.7882 | 0.9283 | **0.9782** | 0.8480 | 2.2 | 253/321 |
| Shiwlu -> español | **0.7913** | **0.9564** | 0.9688 | **0.8617** | **2.0** | **254/321** |

La iteración v3 mejora `R@1` y `MRR` en ambas direcciones. La única caída es
`R@10` Shiwlu -> español, de 0.9720 a 0.9688, equivalente a `-0.31` puntos
porcentuales y dentro del umbral de aceptación definido.

El análisis de errores top-1 reporta 68 errores en español -> Shiwlu y 67 en
Shiwlu -> español. Frente al candidato anterior, esto reduce 12 y 4 errores
top-1 respectivamente. Los errores restantes se concentran en casos con
`gold_has_audit_flag`, `semantic_confusion`, `close_score_ambiguity` y
`shared_shiwilu_tokens`.

### Resumen final de embeddings

- Candidato final provisional: `v3_iterative_hn_e5_base_bidirectional`.
- Razón de aceptación: mejor `R@1` y `MRR` bidireccional frente al candidato
  anterior.
- Riesgo conocido: `R@10` Shiwlu -> español baja levemente de 0.9720 a 0.9688
  (`-0.31` puntos porcentuales), dentro del umbral aceptado.
- Decisión: no hacer más minería iterativa salvo evidencia cualitativa fuerte.
- Siguiente paso: usar este modelo en integración/evaluación con NMT.

El modelo `v3_iterative_hn_e5_base_bidirectional` queda como candidato actual en:

- `models/sentence_transformers/v3_iterative_hn_e5_base_bidirectional`
- `reports/04_embeddings/experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_esp_to_shi_retrieval.json`
- `reports/04_embeddings/experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_shi_to_esp_retrieval.json`
- `reports/04_embeddings/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_training.json`
- `reports/04_embeddings/experiments/v3_iterative_hn_e5_base_bidirectional/v3_iterative_hn_e5_base_bidirectional_freeze_metadata.json`

---

## Salidas por etapa (resumen)

| Etapa | Carpeta | Archivos | Propósito |
|-------|---------|----------|-----------|
| 00 | `data/intermediate/00_pdf/` | `dataset_extraido_pdf.csv` | Pares extraídos del PDF |
| 00 | `reports/00_pdf/` | `summary.json` | Estadísticas de extracción |
| 01 | `data/intermediate/01_filtrado/` | `dataset_filtrado.csv` | Pares válidos de flashcards |
| 01 | `reports/01_filtrado/` | `rows_removed.csv` | Trazabilidad de filas excluidas |
| 01b | `data/intermediate/01b_unificado/` | `dataset_unificado.csv` | **Todas las fuentes combinadas** |
| 01b | `reports/01b_unificado/` | `summary.json`, `cross_duplicates.csv` | Estadísticas y duplicados entre fuentes |
| 02 | `data/intermediate/02_normalizado/` | `dataset_normalizado.csv` | Originales + normalizados |
| 02 | `reports/02_normalizacion/` | `normalization_log.csv`, `rows_removed.csv`, `summary.json` | Bitácora y metadatos |
| 03 | `data/processed/03_pre_embeddings/` | `dataset_pre_embeddings.csv` | **Dataset final para embeddings** |
| 03 | `reports/03_auditoria/` | `problem_rows.csv`, `summary.json` | Problemas y estadísticas |
| 04 | `data/processed/04_splits/` | `train.jsonl`, `valid.jsonl`, `test.jsonl`, `all_text_for_sp.txt` | **Preprocesamiento canónico de embeddings** |
| 04 | `reports/04_embeddings/preprocessing/` | `preprocess_manifest.json`, `preprocessing_closure_report.*` | Manifiesto y cierre del preprocesamiento |
| 05 | `models/sentence_transformers/` | modelos fine-tuned | **Entrenamiento/evaluación de embeddings** |
| 05 | `reports/04_embeddings/{baseline,v1,v2_hn_controlled,experiments/}/` | `*_retrieval.json`, `*_training.json` | Métricas de retrieval y entrenamiento |

---

## Archivos de configuración

### `config/sources.json`

Registro de fuentes de datos. Para agregar una nueva fuente, solo edita este archivo:

```json
{
  "sources": [
    {
      "name": "flashcards",
      "path": "data/intermediate/01_filtrado/dataset_filtrado.csv",
      "esp_column": "ESP",
      "shiwilu_column": "SHIWILU",
      "enabled": true
    },
    {
      "name": "pdf_textos",
      "path": "data/intermediate/00_pdf/dataset_extraido_pdf.csv",
      "esp_column": "ESP",
      "shiwilu_column": "SHIWILU",
      "filter": { "column": "quality_flag", "keep": ["ok", "fallback_last_line_as_spanish"] },
      "enabled": true
    }
  ]
}
```

### `config/normalization_rules.json`

Permite activar/desactivar reglas de normalización sin modificar código.
Incluye placeholders para futuras reglas específicas del shiwilu documentadas
por lingüistas.

---

## Principios de diseño

1. **Organizado por etapa:** Cada paso del pipeline escribe en su propia subcarpeta
2. **Unificación centralizada:** Todas las fuentes convergen en 01b antes de normalización
3. **Trazabilidad:** Toda eliminación o cambio queda registrado con `pair_id` y motivo
4. **No destructivo:** Se preservan columnas originales; normalización en columnas separadas
5. **Configurable:** Fuentes y reglas en JSON externo, fáciles de auditar y modificar
6. **Escalable:** Agregar nuevas fuentes = editar JSON, sin tocar código
7. **Reproducible:** Misma entrada + misma config = misma salida

---

## Próximos pasos

- Usar `v3_iterative_hn_e5_base_bidirectional` como candidato de embeddings para la siguiente fase.
- Revisar manualmente los errores top-1 solo si se quiere depurar datos o `group_id`;
  no iniciar otra ronda de entrenamiento sin esa revisión.
- Preparar integración/evaluación con NMT sin reabrir el preprocesamiento de embeddings.

---

## Pipeline NMT (NLLB + LoRA)

La fase NMT vive en `src/nmt/` y reutiliza el modelo de embeddings cerrado
(`v3_iterative_hn_e5_base_bidirectional`) para filtrado semántico y reranking.
La especificación completa está en [`plan.md`](plan.md) y el plan ejecutable en
`.cursor/plans/sa-binllb_implementation_eb40261e.plan.md`.

### Aislamiento de entorno

El stack NMT usa pines exactos (`requirements/nmt.txt`) y vive en un
entorno virtual aparte para no tocar el entorno de embeddings.

```bash
# Crear entorno aislado (Python 3.12; ver nota más abajo)
python3.12 -m venv .venv-nmt        # o: .conda-emb/bin/python -m venv .venv-nmt
source .venv-nmt/bin/activate
pip install --upgrade pip
pip install -r requirements/nmt.txt

# Verificación
python -c "import torch, transformers, peft, sentence_transformers, faiss, comet, sacrebleu, bert_score; print('ok')"
```

> Nota: `plan.md` sección 6 prescribe Python 3.11; usamos 3.12 porque
> `pyproject.toml` ya lo exige y todos los pines de `requirements/nmt.txt`
> son compatibles con 3.12. Esta desviación está documentada en el cierre
> metodológico de la tesis.

### Estructura adicional para NMT

```
config/nmt/                            # YAMLs por fase (filter, training, inference, reranker, eval)
data/processed/05_nmt_canonical/       # CSV canónico bidireccional (Fase 1)
data/processed/06_nmt_filtered/        # CSV filtrado semánticamente + índices FAISS (Fase 2)
data/processed/07_nmt_augmented/       # backtranslation + minería + variantes morfológicas (Fase 7)
models/nmt/sentencepiece/              # SP Unigram analítico (Fase 3)
models/nmt/tokenizer_shw_extended/     # tokenizer NLLB con shw_Latn (Fase 4a)
models/nmt/nllb_bidi_lora_v0/          # adapters LoRA + tokenizer extendido (Fase 4)
models/nmt/nllb_bidi_lora_v1_bt/       # variante con backtranslation (Fase 7)
reports/05_nmt/{preprocessing,training,evaluation,reranking,augmentation}/
src/nmt/{preprocessing,training,reranking,evaluation,inference,augmentation}/
src/nmt/_legacy/                       # baselines de Transformer-from-scratch (n0-n5)
scripts/nmt/                           # entrypoints por fase (10_*, 20_*, 30_*, ...)
```

### Orden de ejecución (fases)

```bash
# Fase 1: dataset freeze (re-export bidireccional desde data/processed/04_splits/)
python scripts/nmt/10_canonicalize_dataset.py

# Fase 2: filtrado semántico + FAISS (usa el modelo de embeddings v3)
python scripts/nmt/11_semantic_filter.py
python scripts/nmt/12_build_faiss_index.py

# Fase 3: SentencePiece Unigram (artefacto analítico)
python scripts/nmt/13_train_sentencepiece.py

# Fase 4: fine-tuning bidireccional NLLB+LoRA
python scripts/nmt/20_train_nllb_lora.py --config config/nmt/training.yaml

# Fase 5: evaluación completa (BLEU/chrF++/BERTScore/COMET)
python scripts/nmt/30_evaluate.py --checkpoint models/nmt/nllb_bidi_lora_v0 --split test

# Fase 6: reranking semántico (con ablación de pesos)
python scripts/nmt/40_rerank.py --checkpoint models/nmt/nllb_bidi_lora_v0 --split test

# Fase 7: backtranslation + minería (después de un v0 estable)
python scripts/nmt/50_backtranslate.py --checkpoint models/nmt/nllb_bidi_lora_v0
python scripts/nmt/51_train_with_augmented.py --config config/nmt/training.yaml \
    --output models/nmt/nllb_bidi_lora_v1_bt

# Fase 8: evaluación final + comparativa
python scripts/nmt/30_evaluate.py --checkpoint models/nmt/nllb_bidi_lora_v1_bt --split test
python scripts/nmt/40_rerank.py   --checkpoint models/nmt/nllb_bidi_lora_v1_bt --split test
python scripts/nmt/60_compare_runs.py
```

### Modelo base y razones

- Backbone NMT: `facebook/nllb-200-distilled-600M` (multilingual NMT pre-entrenado).
- Adaptación: LoRA (`r=16`, `alpha=32`, `target_modules=["q_proj","v_proj"]`).
  Los 615M parámetros base permanecen congelados.
- Tokenizer: el de NLLB extendido con `shw_Latn` (más `<2shw>` / `<2spa>` por
  compatibilidad con la especificación de `plan.md` §19).
- Inferencia: beam=5, length_penalty=1.0, max_new_tokens=128.
- Reranker: `final = 0.7 * p_translation + 0.3 * cos_sim` con el modelo de
  embeddings v3 ya cerrado.
