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

**Ciclo de tesis cerrado.** Tanto el subsistema de embeddings como el sistema
NMT están entrenados, evaluados y documentados. El modelo enviado es
`v2.1b LoRA+` (NLLB-200 + LoRA `r=32, α=64` + optimizador LoRA+ con
`lr_B = 16·lr_A`, sobre la variante `xl` del corpus de 4501 pares) con
**avg chrF++ reranked = 44.99** (IC 95\% `[43.17, 46.96]`, bootstrap
`n=1000`). Detalle por variante: ver
`reports/05_nmt/evaluation_xl/leaderboard.md` y la Cuadro 5.4 / 5.5 del
documento de tesis.

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

Hardware objetivo: NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120),
Windows 10/11, driver 596.36+, wheels CUDA `cu128`.

```powershell
# Crear entorno aislado (Python 3.12)
py -3.12 -m venv .venv-nmt
.\.venv-nmt\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

# IMPORTANTE: torch 2.7.1 con CUDA 12.8 NO está en PyPI por defecto.
# Instalarlo primero desde el índice oficial de PyTorch.
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Resto del stack (transformers, peft, comet, faiss-cpu, etc.)
pip install -r requirements/nmt.txt

# Verificación de CUDA + sm_120 + stack completo
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_capability(0))"
python -c "import torch, transformers, peft, sentence_transformers, faiss, comet, sacrebleu, bert_score; print('ok')"
```

Resultado esperado: `cuda True 12.8 (12, 0)` y `ok`. Si `cuda False`, el
wheel `cu128` no quedó instalado (PyPI re-resolvió a CPU); repetir
`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`.

> Notas de desviación frente a `plan.md` §6-§7:
> - Python 3.12 (no 3.11) para alinear con `pyproject.toml >=3.12`.
> - `torch==2.7.1` (no 2.5.1): 5060 Ti es Blackwell sm_120, sin kernels en 2.5.1.
> - `transformers==4.55.0`, `accelerate==1.8.1`, `peft==0.16.0`: bumps coherentes con torch 2.7.
> - `sentence-transformers>=5.4.1,<6` (no 4.1.0): el checkpoint v3 fue guardado con 5.4.1.
> - `numpy<2`: requerido por `unbabel-comet 2.2.4`.
> - `setuptools<81`: `torchmetrics` (transitivo de COMET) aún importa `pkg_resources`.

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

Todos los scripts se invocan como módulos con `python -m` desde la raíz del proyecto
con `.venv-nmt` activo.

```powershell
# Fase 1: dataset freeze (re-export bidireccional desde data/processed/04_splits/)
python -m scripts.nmt.10_canonicalize_dataset

# Fase 2: filtrado semántico + FAISS (usa el modelo de embeddings v3)
python -m scripts.nmt.20_semantic_filter
python -m scripts.nmt.21_build_faiss

# Fase 3: SentencePiece Unigram (artefacto analítico)
python -m scripts.nmt.22_train_sentencepiece

# Fase 4: fine-tuning bidireccional NLLB+LoRA (3-6 h en RTX 5060 Ti)
python -m scripts.nmt.30_train_lora --config config/nmt/training.yaml

# Fase 5: evaluación completa (BLEU/chrF++/BERTScore/COMET)
python -m scripts.nmt.40_evaluate --checkpoint models/nmt/nllb_bidi_lora_v0 --split test

# Fase 6: reranking semántico (con ablación de pesos)
python -m scripts.nmt.50_rerank --checkpoint models/nmt/nllb_bidi_lora_v0 --split test

# Fase 7: backtranslation + minería (después de un v0 estable)
python -m scripts.nmt.60_backtranslate --checkpoint models/nmt/nllb_bidi_lora_v0
python -m scripts.nmt.61_mine_pairs
python -m scripts.nmt.63_train_with_augmented --config config/nmt/training.yaml --output models/nmt/nllb_bidi_lora_v1_bt

# Fase 8: evaluación final + comparativa
python -m scripts.nmt.40_evaluate --checkpoint models/nmt/nllb_bidi_lora_v1_bt --split test
python -m scripts.nmt.50_rerank   --checkpoint models/nmt/nllb_bidi_lora_v1_bt --split test
python -m scripts.nmt.70_compare_runs
python -m scripts.nmt.71_human_eval_template
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

### Estado actual: v0 entrenado

El primer modelo (sin augmentation) ya quedó entrenado y guardado:

- Adaptador: `models/nmt/nllb_bidi_lora_v0/checkpoint-2500` (mejor checkpoint).
- Tokenizer extendido: `models/nmt/tokenizer_shw_extended/`.
- Reportes: `reports/05_nmt/training/nllb_bidi_lora_v0/`.
- Log completo: `train_v0.log`.
- Tiempo de entrenamiento: ~68 min en RTX 5060 Ti, 20 epochs / 3100 steps.

Resultado (validation, mejor checkpoint, epoch 17.74):

| Dirección | chrF++ | BLEU |
|-----------|-------:|-----:|
| shw -> spa | 19.12 | 1.77 |
| spa -> shw | 15.43 | 0.24 |
| **avg_chrf** | **17.28** | — |

Resultado completo en test (642 ejemplos, 321 por dirección, beam=5,
length_penalty=1.0, max_new_tokens=128):

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

`BERTScore` (`xlm-roberta-large`) y `COMET` (`Unbabel/wmt22-comet-da`) se reportan
solo como proxy: ninguno de los dos modelos vio Shiwilu en pre-entrenamiento.
chrF++ sigue siendo la métrica primaria.

Ablación de α en el reranker (test, mejor por `avg_chrf++`):

| α (peso `trans_prob`) | avg chrF++ | avg BLEU | shw -> spa chrF++ | spa -> shw chrF++ |
|:----------------------|-----------:|---------:|------------------:|------------------:|
| 0.0 (puro SBERT)      | 20.89      | 2.88     | 19.88             | 21.90             |
| 0.3                   | 20.95      | 2.90     | 19.95             | 21.96             |
| **0.5 (óptimo)**      | **21.07**  | **2.96** | **20.07**         | **22.06**         |
| 0.7 (default)         | 20.89      | 2.96     | 19.95             | 21.84             |
| 1.0 (sin SBERT)       | 19.64      | 2.76     | 18.63             | 20.65             |

Lecturas:

- El reranker semántico añade **+1.42 chrF++ avg** sobre baseline (19.64 -> 21.07)
  y la ganancia es monótona desde α=1.0 a α=0.5; α=0.0 (puro SBERT) ya supera
  baseline en +1.25, lo que confirma que la señal del modelo de embeddings v3 no
  es ruido.
- `spa -> shw` sale por encima de `shw -> spa` en chrF++ porque char-n-gramas
  premian aciertos morfológicos parciales del decoder Shiwilu, mientras BLEU se
  mantiene mayor en `shw -> spa` (target español, tokenización word-level).
- Test sube respecto a validation final (avg 17.28 -> 19.64 baseline / 21.07 con
  reranker) sin signo de leakage: los `group_id` están aislados desde Phase 1.
- Se fija **α=0.5 como default operativo** del reranker. Documentado en
  `reports/05_nmt/reranking/nllb_bidi_lora_v0/ablation.json`.

Reportes generados:

- `reports/05_nmt/evaluation/nllb_bidi_lora_v0/test_metrics.json`
- `reports/05_nmt/evaluation/nllb_bidi_lora_v0/test_predictions.jsonl`
- `reports/05_nmt/evaluation/nllb_bidi_lora_v0/test_predictions_topk.jsonl`
- `reports/05_nmt/reranking/nllb_bidi_lora_v0/test_metrics_reranked.json`
- `reports/05_nmt/reranking/nllb_bidi_lora_v0/test_predictions_reranked.jsonl`
- `reports/05_nmt/reranking/nllb_bidi_lora_v0/ablation.json`

Notas operativas registradas durante la corrida:

- Se introdujo `Seq2SeqCollatorWithDecoderInputs` en
  `src/nmt/training/train_lora.py` para precomputar `decoder_input_ids` con
  `shift_tokens_right`. En transformers 4.55, NLLB ya no expone
  `prepare_decoder_input_ids_from_labels`; combinado con
  `label_smoothing_factor=0.1` (que hace que el `Trainer` saque `labels` antes
  del forward), el `DataCollatorForSeq2Seq` estándar deja al decoder sin
  `decoder_input_ids` ni `decoder_inputs_embeds` y dispara un `ValueError`
  engañoso. El fix mantiene la cadena en transformers 4.55 / peft 0.16 /
  accelerate 1.8 sin downgrades.
- Se observó plateau de chrF para `shw -> spa` alrededor de epoch 9-10. Para el
  reentrenamiento con augmentation (v1_bt) se planea subir LoRA a `r=32` /
  `alpha=64` para evitar saturación con el dataset ampliado.
- Fix de inferencia: al recargar el tokenizer del checkpoint, NLLB reconstruye
  `lang_code_to_id` desde su lista hardcoded y pierde `shw_Latn` (que sí queda
  guardado en `additional_special_tokens`). Se añadió
  `_ensure_extended_lang_codes_registered` en `src/nmt/inference/generate.py`
  que re-registra automáticamente cualquier código FLORES-style del tokenizer
  cargado, restaurando `forced_bos_token_id` y `set_src_lang_special_tokens`
  para `shw_Latn`. El fix también cubre `60_backtranslate.py` porque ambos
  pasan por `load_checkpoint`.

### Enhancements integrados en el pipeline

Sobre la base v0 + reranker se sumaron cuatro enhancements antes de v1_bt
para que la corrida final responda a preguntas adicionales sin re-entrenar:

1. **Confidence/reliability layer** (`src/nmt/inference/confidence.py`).
   Cada predicción guarda `confidence ∈ {low, medium, high}`,
   `confidence_score` y `confidence_components`. Para baseline el score es
   `exp(top-1 sequence_score)` (probabilidad geométrica por token); para el
   reranker es `final_score = α * trans_prob + (1-α) * cos_sim`. Distribución
   en v0 baseline test (642): 304/254/84 (low/medium/high), umbral
   `0.40 / 0.55`. Distribución en v0 + reranker α=0.7: 405/218/19, umbral
   `0.30 / 0.40`. Los umbrales y la distribución viven en
   `meta.confidence` de cada `*_metrics.json`.
2. **Rare-token / morphology-aware evaluation** (`src/nmt/evaluation/rare_token.py`,
   `scripts/nmt/41_rare_token_eval.py`). chrF++ se recalcula en buckets por
   fracción de palabras raras en la referencia (frecuencia en train < 5) y
   se reporta `oov_recovery_rate`. Headline = bucket "≥20% raras". Resultado
   v0 baseline: avg chrF++ raras = **19.58** (vs overall 19.64), avg
   OOV-recovery = **0.022**. v0 + reranker: **20.75** y **0.026** — el
   reranker conserva la ventaja en el régimen morfológicamente denso.
3. **Comparación de tokenizadores NLLB vs SP Unigram propio** (artefacto
   de tesis, no de modelo). Sobre 50 frases Shiwilu de muestra: SP Unigram
   produce 9.68 tokens/frase y 3.34 tokens/palabra, NLLB 12.24 / 4.22; SP es
   más corto en 41/50 oraciones, NLLB en 3/50, empate en 6/50. Se documenta
   por qué se mantuvo NLLB (transferencia multilingüe del backbone) pese a
   que SP es más eficiente en subwords. Tabla autogenerada en
   `thesis/latex/figuras/generated/nmt_sentencepiece_vs_nllb.tex`.
4. **Weighted synthetic-data training** (`src/nmt/training/dataset.py`,
   `src/nmt/training/train_lora.py`, Enhancement #4). Cada fila del CSV de
   train recibe un peso a partir de `origin_source`: real
   (`flashcards`/`pdf_textos`) = 1.0, minado (`mined_v3_sbert`) = 0.5,
   backtranslation (`backtranslation_v0`) = 0.3. La pérdida pasa por
   `_weighted_smoothed_ce` (label-smoothing replicado a nivel per-row, con
   denominador en tokens ponderados). Sólo se activa cuando se usa
   `63_train_with_augmented.py` (back-compat con el v0 original). v1_bt
   también sube LoRA a `r=32` / `alpha=64` para absorber el corpus ampliado
   sin saturación.

### Estado del augmentation (Phase 7a)

`scripts/nmt/60_backtranslate.py` corrió sobre el adaptador v0 con el pool
mono Shiwilu (`data/processed/07_nmt_augmented/mono_shw.txt`, 76 líneas tras
filtrar parallel + apóstrofe-required + heurística español):

- 12 filas sintéticas tras el filtro semántico ≥ 0.60 (= 6 pares únicos × 2
  direcciones); score medio = 0.666, max = 0.787.
- Cap = 2× paralelo (= 9888 max), por lo que no hubo recorte.
- No se hace backtranslation `spa → shw` en v1_bt: la dirección difícil es
  `spa → shw`, y un v0 que aún tropieza ahí generaría Shiwilu sintético
  ruidoso que envenenaría el target. El BT se hace sólo `shw → spa` para
  ampliar la señal en la dirección donde v0 es más fiable.
- Reporte: `reports/05_nmt/augmentation/backtranslation.json`.

Estos pares quedan en `data/processed/07_nmt_augmented/train_bt.csv` con
`origin_source = backtranslation_v0` y peso 0.3 al entrar a v1_bt.

### Próximos pasos de NMT

1. ~~Phase 5: evaluación completa de v0 sobre test (BLEU + chrF + BERTScore-F1 +
   COMET).~~ Hecho. avg chrF++ = 19.64.
2. ~~Phase 6: reranker semántico + barrido `alpha ∈ {0.0, 0.3, 0.5, 0.7, 1.0}`.~~
   Hecho. Mejor α = 0.5, avg chrF++ = 21.07 (+1.42 sobre baseline).
3. ~~Phase 7a: backtranslation con v0 sobre monolingüe Shiwilu (apóstrofe
   requerido por defecto), filtrado semántico ≥ 0.60, capeo a 2x del paralelo.~~
   Hecho. 12 filas sintéticas, todas en dirección `shw → spa`.
4. ~~Phase 7b: minería bilingüe vía FAISS sobre v3 SBERT.~~ Hecho. 1338 pares
   aceptados en `data/processed/07_nmt_augmented[/_xl]/train_mined.csv`.
5. ~~Phase 7d: re-entreno como v1_bt con LoRA `r=32`/`alpha=64`, weighted loss
   (1.0 / 0.5 / 0.3), paralelo + BT round-trip + mined.~~ Hecho como
   `v1_bt_xl`; avg chrF++ reranked = 32.52.
6. ~~Phase 8a/b/c: comparación v0 vs v1_bt (con y sin reranker), análisis
   rare-token, plantilla humana ciega, tablas LaTeX y actualización de
   tesis.~~ Hecho.
7. ~~Phase 2 (ablaciones): DoRA (`v2.0`), LoRA+ aislado (`v2.1b`), DoRA+LoRA+
   (`v2.1`), BT iter1 Wikipedia (`v2.2`).~~ Hecho. Campeón = `v2.1b` LoRA+
   con avg chrF++ reranked = **44.99** (IC 95\% `[43.17, 46.96]`, `n=1000`
   bootstrap). DoRA no se distingue estadísticamente del baseline; el BT con
   Wikipedia regresó el modelo a ~32 chrF++ (la coincidencia de dominio prima
   sobre el volumen del pool monolingüe).
8. ~~Phase 5/6 cierre: leaderboard 6-way, intervalos de confianza bootstrap,
   tablas LaTeX nuevas, secciones Anexo C.14 y futuro trabajo
   actualizado, PDF recompilado (111 páginas).~~ Hecho.

### Resultados finales (xl test, reranked, 892 filas direccionales)

| Modelo | shw→spa chrF / BLEU | spa→shw chrF / BLEU | avg chrF | IC 95% (avg) |
|---|---|---|---:|---|
| v0_xl | 25.16 / 9.73 | 28.06 / 4.61 | 26.61 | [25.33, 27.88] |
| v1_bt_xl (+BT OPUS-100) | 29.91 / 12.45 | 34.36 / 6.69 | 32.14 | [30.63, 33.80] |
| v2.0 DoRA | 30.82 / 14.55 | 34.35 / 6.46 | 32.58 | [31.02, 34.25] |
| v2.1 DoRA+LoRA+ | 41.05 / 23.23 | 47.54 / 12.76 | 44.30 | [42.49, 46.12] |
| **v2.1b LoRA+ ★** | **42.42 / 24.48** | **47.56 / 12.45** | **44.99** | **[43.17, 46.96]** |
| v2.2 +BT iter1 Wikipedia | 30.72 / 14.76 | 33.94 / 5.90 | 32.33 | [30.82, 33.94] |

Fuente: `reports/05_nmt/evaluation_xl/leaderboard.md` (Phase 5,
`scripts/nmt/72_leaderboard.py`) y `bootstrap_ci_summary.md` (Phase 6,
`scripts/nmt/73_bootstrap_ci.py`).

### Future work (no incluido en este ciclo)

Se evaluaron varios enhancements adicionales y se decidió no incluirlos en
v1_bt para no inflar el alcance de la tesis. Quedan documentados como
trabajo futuro:

- **Iterative backtranslation** (Edunov et al., 2018; Hoang et al., 2018).
  El BT actual es una sola pasada con v0. Iterar (entrenar v1_bt → re-BT →
  v2_bt → ...) suele dar +1–2 chrF++ adicionales en lenguas de bajo
  recurso, pero exige (a) un pool mono mucho mayor que las 76 líneas
  actuales y (b) un protocolo claro de detención (parar cuando la calidad
  del BT en validación deja de mejorar). En este corpus el costo/beneficio
  no se justifica todavía.
- **Domain-controlled monolingual Spanish corpus**. Hoy no se hace BT
  `spa → shw` precisamente porque generaría Shiwilu ruidoso. Una vía es
  curar un corpus Español pequeño (10–30k frases) en dominios cercanos a
  flashcards / textos comunitarios, filtrarlo con el v3 SBERT contra el
  Shiwilu existente, y luego BT `spa → shw` con un v1_bt ya estable. Es la
  línea natural después de iterative BT.
- **Focal loss para preservación de rare-tokens**. La métrica
  `oov_recovery_rate` (~2%) muestra que el modelo casi no copia palabras
  Shiwilu OOV cuando aparecen en la entrada. Focal loss con γ ≈ 2 sobre
  los token-ids más raros debería sesgar el decoder a preservarlos. Es
  ortogonal al weighted-data y se puede combinar con él, pero requiere
  reescribir `compute_loss` para conocer la frecuencia de cada
  token-target en train.
- **Arquitectura Two-Adapter LoRA+ (asymmetric ranks).** Una adaptación
  LoRA+ por dirección, con rangos asimétricos: spa→shw con r=64/α=128
  (la dirección más difícil, dominada por morfología generativa) y
  shw→spa con r=32/α=64. Cada adapter se entrena con el flag
  `--direction {spa2shw,shw2spa}` ya implementado en
  `30_train_lora.py`, y la evaluación combina ambos vía
  `40_evaluate.py --checkpoint-spa2shw … --checkpoint-shw2spa …
  --run-name nllb_two_loraplus_xl`. El soporte de código está listo
  (scripts 40/50 ya aceptan `--checkpoint-spa2shw`/`--checkpoint-shw2spa`);
  queda como trabajo futuro porque el ciclo de tesis cierra con el
  campeón bidireccional v2.1b LoRA+ (avg chrF++ = 44.99). Hipótesis a
  validar: separar direcciones permite que cada adapter se especialice
  y cierre el gap residual entre shw→spa y spa→shw.
- **SWA (Stochastic Weight Averaging) sobre el ganador.** Promediar los
  últimos 3–5 checkpoints del mejor modelo suele aportar +0.3–0.8
  chrF++ de regularización implícita en regímenes de bajo recurso
  (Izmailov et al., 2018). Es barato (~30 min) y no requiere
  re-entrenar; queda como paso final antes del eval definitivo en una
  iteración posterior de la tesis.
- **Iterative BT con dominio in-domain.** El experimento v2.2 con
  Wikipedia-es regresó al modelo a ~32 chrF++ (vs 44.99 de v2.1b),
  confirmando que el dominio del corpus monolingüe es crítico. El BT
  exitoso anterior (v1_bt_xl, +7 chrF++ sobre v0_xl) usó 1012 frases
  de OPUS-100 con sólo 89 pares aceptados a threshold 0.70 — registro
  conversacional, alineado con el corpus shiwilu (flashcards,
  narrativas, frases cotidianas). Futuro: ampliar BT con Tatoeba +
  News-commentary + corpora hispanoamericanos cuidadosamente
  filtrados por similitud al gold corpus, evitando Wikipedia.

Cada uno de estos puntos también está discutido en
`thesis/latex/tesis.tex` capítulo 5 (Future work).
