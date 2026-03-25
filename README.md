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
├── data/
│   ├── raw/                 # Datos originales sin modificar
│   │   ├── flashcards2.csv
│   │   └── dataset_esp_shiwilu.csv
│   └── processed/           # Datos limpios (salida de los pipelines)
│       └── dataset_limpio.csv
├── scripts/                 # Pipelines de procesamiento
│   ├── 01_filtrar_dataset.py
│   └── 02_depurar_dataset.py
├── models/                  # Modelos entrenados (siguiente fase)
├── notebooks/               # Exploración interactiva (siguiente fase)
├── reports/                 # Gráficas generadas por los pipelines
├── pyproject.toml           # Dependencias del proyecto (Poetry)
├── poetry.lock              # Versiones exactas de todas las dependencias
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
poetry run python scripts/02_depurar_dataset.py
```

### Opción B: Activar el entorno una vez y ejecutar sin prefijo

```cmd
.venv\Scripts\activate
```

Verás que la terminal muestra `(.venv)` al inicio de la línea. Desde ese
momento puedes ejecutar comandos directamente sin el prefijo `poetry run`:

```cmd
python scripts/02_depurar_dataset.py
```

Para desactivar el entorno cuando termines:

```cmd
deactivate
```

---

## Ejecución de pipelines

### Pipeline 01: Filtrar dataset

Extrae del CSV original solo las filas que tienen valores válidos en ambas
columnas (ESP y SHIWILU), descartando filas vacías o sin traducción.

```cmd
poetry run python scripts/01_filtrar_dataset.py
```

**Entrada:**

- `data/raw/flashcards2.csv`

**Salida:**

- `data/raw/dataset_esp_shiwilu.csv`

### Pipeline 02: Depurar dataset

Ejecuta un pipeline de 7 pasos de limpieza sobre el dataset filtrado y genera
un reporte en consola junto con 4 gráficas de análisis.

```cmd
poetry run python scripts/02_depurar_dataset.py
```

**Entrada:**

- `data/raw/dataset_esp_shiwilu.csv`

**Salida:**

- `data/processed/dataset_limpio.csv`
- `reports/distribucion_longitudes.png`
- `reports/top_palabras.png`
- `reports/comparacion_antes_despues.png`
- `reports/longitud_correlacion.png`

---

## Pipeline de depuración: detalle de los 7 pasos

| Paso | Descripción |
|------|-------------|
| 1    | Trim y normalización de espacios (trailing, dobles, comas con espacio extra) |
| 2    | Conversión a minúsculas en ambas columnas |
| 3    | Eliminación de signos de puntuación (preserva apóstrofos del shiwilu) |
| 4    | Limpieza de paréntesis explicativos en español |
| 5    | Normalización Unicode NFC |
| 6    | Eliminación de duplicados exactos |
| 7    | Exportación del CSV limpio, reporte en consola y generación de gráficas |

---

## Próximos pasos

- Generación de embeddings bilingües (FastText / fine-tuning XLM-RoBERTa)
- Entrenamiento del modelo NMT
- Evaluación con métricas BLEU, chrF y evaluación humana
