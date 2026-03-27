# NMT Español-Shiwilu (spa-jeb)

Sistema de Traducción Automática Neuronal (Neural Machine Translation) entre
español (`spa`) y shiwilu/jebero (`jeb`), una lengua originaria de la familia
cahuapana hablada en la Amazonía peruana.

El shiwilu es una lengua de bajos recursos con escasos datos digitalizados.
Este proyecto aborda el desafío de construir un sistema de traducción automática
a partir de un corpus paralelo limitado, aplicando técnicas de preprocesamiento,
generación de embeddings bilingües y entrenamiento de modelos neuronales.

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
│       └── 03_pre_embeddings/
│           └── dataset_pre_embeddings.csv
├── scripts/
│   ├── 00_extraer_dataset_pdf.py       # Etapa 0: Extracción desde PDF
│   ├── 01_filtrar_dataset.py           # Etapa 1: Filtrado inicial (flashcards)
│   ├── 01b_unificar_fuentes.py         # Etapa 1b: Unificación de fuentes
│   ├── 02_depurar_dataset.py           # Etapa 2: Normalización no destructiva
│   └── 03_auditar_dataset.py           # Etapa 3: Auditoría y exportación final
├── src/
│   └── embeddings/
│       ├── preprocess.py               # Tokenización del corpus
│       ├── train_fasttext.py           # Etapa 4: Entrenamiento FastText
│       ├── train_sentence_transformers.py  # Etapa 4b: Sentence Transformers
│       ├── compare_embeddings.py       # Etapa 5: Comparación de embeddings
│       └── utils.py                    # Utilidades para embeddings
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
│       ├── similarity_scores.csv       # Scores de similitud cross-lingual
│       ├── comparison_report.json      # Comparación FastText vs ST
│       └── low_similarity_pairs.csv    # Pares candidatos a filtrar
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

## Etapa 04: Entrenamiento de embeddings FastText

**Script:** `src/embeddings/train_fasttext.py`

Entrena embeddings FastText (Skip-Gram) desde cero usando el corpus bilingüe.
Concatena español y shiwilu en un único espacio de vectores, lo cual es ideal
para lenguas de bajos recursos con morfología rica.

**Entrada:**
- `data/processed/03_pre_embeddings/dataset_pre_embeddings.csv`

**Salidas:**
- `models/fasttext/fasttext.model` - Modelo completo (recargable con gensim)
- `models/fasttext/fasttext.vec` - Vectores en formato word2vec (portable)

**Ejecución:**

```cmd
poetry run python src/embeddings/train_fasttext.py
```

**Opciones:**

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--data` | `data/processed/03_pre_embeddings/dataset_pre_embeddings.csv` | Ruta al CSV |
| `--vector-size` | 100 | Dimensión de los vectores |
| `--window` | 5 | Tamaño de ventana de contexto |
| `--min-count` | 1 | Frecuencia mínima (1 = incluir todas) |
| `--epochs` | 5 | Iteraciones de entrenamiento |

**Configuración por defecto (Skip-Gram):**
- `vector_size=100`: Dimensión adecuada para corpus pequeño
- `window=5`: Contexto local para capturar relaciones sintácticas
- `min_count=1`: Incluye todas las palabras (importante para low-resource)
- `sg=1`: Skip-Gram (mejor para palabras raras que CBOW)

**Uso de los embeddings:**

```python
from gensim.models import FastText

model = FastText.load("models/fasttext/fasttext.model")

# Obtener vector (funciona para OOV gracias a subpalabras)
vector = model.wv["palabra"]

# Palabras similares
similares = model.wv.most_similar("hola", topn=10)
```

---

## Etapa 04b: Embeddings Sentence Transformers

**Script:** `src/embeddings/train_sentence_transformers.py`

Genera embeddings a nivel de oración usando un modelo multilingüe pre-entrenado.
Permite medir similitud cross-lingual entre pares español-shiwilu.

**Entrada:**
- `data/processed/03_pre_embeddings/dataset_pre_embeddings.csv`

**Salidas:**
- `models/sentence_transformers/embeddings_esp.npy` - Embeddings oraciones español
- `models/sentence_transformers/embeddings_shi.npy` - Embeddings oraciones shiwilu
- `reports/04_embeddings/similarity_scores.csv` - Scores de similitud por par
- `reports/04_embeddings/sentence_transformers_summary.json` - Estadísticas

**Ejecución:**

```cmd
poetry run python src/embeddings/train_sentence_transformers.py
```

**Modelo usado:** `intfloat/multilingual-e5-small` (pre-entrenado multilingüe)

**Uso de los scores:**
- Pares con baja similitud son candidatos a revisar/filtrar
- Los scores ayudan a identificar problemas en el corpus

---

## Etapa 05: Comparación de Embeddings

**Script:** `src/embeddings/compare_embeddings.py`

Compara FastText vs Sentence Transformers para analizar las diferencias en cómo
cada método representa el corpus bilingüe.

**Entrada:**
- `models/fasttext/fasttext.model`
- `models/sentence_transformers/embeddings_*.npy`
- `reports/04_embeddings/similarity_scores.csv`

**Salidas:**
- `reports/04_embeddings/comparison_report.json` - Reporte comparativo
- `reports/04_embeddings/low_similarity_pairs.csv` - Pares problemáticos

**Ejecución:**

```cmd
poetry run python src/embeddings/compare_embeddings.py
```

**Interpretación:**
- FastText captura similitud **morfológica** (subpalabras, estructura interna)
- Sentence Transformers captura similitud **semántica** cross-lingual
- Ambos métodos son complementarios para el pipeline NMT

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
| 04 | `models/fasttext/` | `fasttext.model`, `fasttext.vec` | **Embeddings FastText (Skip-Gram)** |
| 04b | `models/sentence_transformers/` | `embeddings_esp.npy`, `embeddings_shi.npy` | **Embeddings Sentence Transformers** |
| 04b | `reports/04_embeddings/` | `similarity_scores.csv`, `*_summary.json` | Scores de similitud cross-lingual |
| 05 | `reports/04_embeddings/` | `comparison_report.json`, `low_similarity_pairs.csv` | Comparación y pares problemáticos |

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

- Revisión manual de `reports/03_auditoria/problem_rows.csv` para decidir exclusiones
- Comparación de embeddings: FastText vs Word2Vec
- Embeddings contextuales: fine-tuning XLM-RoBERTa
- Entrenamiento del modelo NMT
- Evaluación con métricas BLEU, chrF y evaluación humana
