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
│   └── normalization_rules.json    # Reglas de normalización configurables
├── data/
│   ├── raw/                        # Datos originales sin modificar
│   │   └── flashcards2.csv
│   ├── intermediate/               # Datos en proceso (entre etapas)
│   │   ├── dataset_filtrado.csv
│   │   └── dataset_auditado.csv
│   └── processed/                  # Datos finales listos para modelos
│       └── dataset_pre_embeddings.csv
├── scripts/
│   ├── 01_filtrar_dataset.py       # Etapa 1: Filtrado inicial
│   ├── 02_depurar_dataset.py       # Etapa 2: Normalización no destructiva
│   └── 03_auditar_dataset.py       # Etapa 3: Auditoría y exportación final
├── models/                         # Modelos entrenados (siguiente fase)
├── notebooks/                      # Exploración interactiva (siguiente fase)
├── reports/                        # Reportes y bitácoras de preprocesamiento
│   ├── rows_removed_01_filtrado.csv
│   ├── normalization_log.csv
│   ├── rows_removed_02_depuracion.csv
│   ├── preprocessing_summary.json
│   ├── audit_problem_rows.csv
│   └── audit_summary.json
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

El pipeline tiene 3 etapas secuenciales. Cada etapa genera salidas trazables
y no destruye información de las etapas anteriores.

### Orden de ejecución

```cmd
poetry run python scripts/01_filtrar_dataset.py
poetry run python scripts/02_depurar_dataset.py
poetry run python scripts/03_auditar_dataset.py
```

---

## Etapa 01: Filtrado inicial

**Script:** `scripts/01_filtrar_dataset.py`

Filtra el CSV original para quedarse solo con filas que tengan valores válidos
en ambas columnas (ESP y SHIWILU). Asigna un `pair_id` único a cada par.

**Entrada:**
- `data/raw/flashcards2.csv`

**Salidas:**
- `data/intermediate/dataset_filtrado.csv` — Dataset con columnas `pair_id`, `ESP`, `SHIWILU`
- `reports/rows_removed_01_filtrado.csv` — Log de filas removidas con motivo

**Criterios de exclusión:**
- Filas con `ESP` o `SHIWILU` vacío, nulo, o placeholder `"--"`

---

## Etapa 02: Normalización no destructiva

**Script:** `scripts/02_depurar_dataset.py`

Aplica normalización configurable sin destruir el texto original. Mantiene
columnas separadas para texto original y normalizado.

**Entrada:**
- `data/intermediate/dataset_filtrado.csv`
- `config/normalization_rules.json`

**Salidas:**
- `data/intermediate/dataset_auditado.csv` — Dataset con columnas:
  - `pair_id`
  - `ESP_original`, `SHIWILU_original`
  - `ESP_normalizado`, `SHIWILU_normalizado`
- `reports/normalization_log.csv` — Log granular de cada transformación aplicada
- `reports/rows_removed_02_depuracion.csv` — Log de filas removidas (vacío por defecto)
- `reports/preprocessing_summary.json` — Metadatos y estadísticas de la corrida

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
- `data/intermediate/dataset_auditado.csv`

**Salidas:**
- `reports/audit_problem_rows.csv` — CSV con filas problemáticas consolidadas
- `reports/audit_summary.json` — Resumen JSON de auditoría completo
- `data/processed/dataset_pre_embeddings.csv` — Dataset final con columna `has_audit_flags`

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

## Archivo de configuración

**Ubicación:** `config/normalization_rules.json`

Permite activar/desactivar reglas de normalización sin modificar código.
Incluye placeholders para futuras reglas específicas del shiwilu documentadas
por lingüistas.

Ejemplo de estructura:

```json
{
  "global_rules": {
    "lowercase": {
      "enabled": true,
      "description": "Convertir a minúsculas para normalización",
      "order": 5
    }
  },
  "language_specific": {
    "SHIWILU": {
      "orthographic_variants": {
        "enabled": false,
        "description": "Placeholder para variantes ortográficas"
      }
    }
  }
}
```

---

## Salidas por etapa (resumen)

| Etapa | Archivo | Propósito |
|-------|---------|-----------|
| 01 | `data/intermediate/dataset_filtrado.csv` | Pares válidos con pair_id |
| 01 | `reports/rows_removed_01_filtrado.csv` | Trazabilidad de filas excluidas |
| 02 | `data/intermediate/dataset_auditado.csv` | Originales + normalizados |
| 02 | `reports/normalization_log.csv` | Bitácora de transformaciones |
| 02 | `reports/preprocessing_summary.json` | Metadatos de la corrida |
| 03 | `reports/audit_problem_rows.csv` | Filas con problemas detectados |
| 03 | `reports/audit_summary.json` | Estadísticas y vocabulario |
| 03 | `data/processed/dataset_pre_embeddings.csv` | **Dataset final para embeddings** |

---

## Principios de diseño

1. **Trazabilidad:** Toda eliminación o cambio queda registrado con `pair_id` y motivo
2. **No destructivo:** Se preservan columnas originales; normalización en columnas separadas
3. **Configurable:** Reglas en JSON externo, fáciles de auditar y modificar
4. **Conservador:** Reglas agresivas desactivadas por defecto para no perder información
5. **Reproducible:** Misma entrada + misma config = misma salida

---

## Próximos pasos

- Revisión manual de `audit_problem_rows.csv` para decidir exclusiones
- Generación de embeddings bilingües (FastText / fine-tuning XLM-RoBERTa)
- Entrenamiento del modelo NMT
- Evaluación con métricas BLEU, chrF y evaluación humana
