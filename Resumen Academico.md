# Resumen Academico

## Idea central

Esta tesis construye un sistema de traduccion automatica neuronal entre shiwilu y castellano, una lengua de bajos recursos. Como no existe mucho corpus digital limpio, el trabajo no consiste solo en entrenar un modelo: primero se crea y depura un corpus paralelo, luego se entrena un modelo de embeddings bilingue para medir similitud semantica, despues se adapta un modelo NMT bidireccional y finalmente se prepara un protocolo de validacion automatica y participativa.

Flujo general:

```text
Datos crudos
-> limpieza y normalizacion
-> corpus paralelo
-> embeddings bilingues
-> filtrado semantico / apoyo al NMT
-> modelo NMT shiwilu <-> castellano
-> evaluacion automatica
-> protocolo de validacion humana preparado
```

## 1. Limpieza y preparacion de datos

La primera parte del codigo trabaja con datos crudos ubicados en `data/raw/`. Las fuentes incluyen flashcards, textos bilingues y material extraido desde PDF.

Scripts principales:

- `scripts/00_extraer_dataset_pdf.py`
- `scripts/01_filtrar_dataset.py`
- `scripts/02_depurar_dataset.py`
- `scripts/03_auditar_dataset.py`
- `src/embeddings/preprocess_embeddings.py`
- `src/embeddings/audit_preprocessing.py`

El objetivo fue convertir fuentes heterogeneas en un corpus paralelo usable. La limpieza fue no destructiva, porque en shiwilu no conviene aplicar reglas agresivas pensadas para castellano. No se deben corregir sufijos, cortar morfologia ni normalizar como si todo fuera espanol.

Regla central:

```text
limpiar ruido, pero no alterar rasgos linguisticos del shiwilu
```

Se eliminaron duplicados exactos, filas vacias, pares problematicos y registros que no servian como traduccion paralela. Cada decision quedo respaldada por reportes:

- `reports/01_filtrado/`
- `reports/02_normalizacion/`
- `reports/03_auditoria/`

El resultado fue un dataset limpio, auditado y dividido en particiones reproducibles de entrenamiento, validacion y prueba.

## 2. Por que usamos embeddings

Despues de limpiar los datos, se necesitaba una forma de medir si una frase en castellano y una frase en shiwilu estaban semanticamente cerca. Para eso se usaron embeddings bilingues.

Un embedding convierte una oracion en un vector numerico. Si dos oraciones significan algo parecido, sus vectores deberian quedar cerca en el espacio.

En esta tesis los embeddings sirven para:

- evaluar recuperacion bilingue;
- detectar pares dudosos;
- filtrar datos sinteticos;
- apoyar el reranking del traductor;
- tener una senal semantica independiente del modelo NMT.

El modelo base elegido fue:

```text
intfloat/multilingual-e5-base
```

La variante final fue:

```text
v3_iterative_hn_e5_base_bidirectional
```

Se uso E5 porque es un modelo multilingue fuerte para recuperacion semantica. En vez de entrenar embeddings desde cero con pocos datos, se aprovecho un modelo preentrenado y se adapto al par shiwilu-castellano.

## 3. Ajuste de los embeddings

El modelo de embeddings no se uso directamente en su version base. Se siguio una progresion experimental:

```text
E5-base sin fine-tuning
-> fine-tuning bidireccional
-> hard negative mining controlado
-> iterative hard negative mining v3
```

La idea de los hard negatives es importante. Un ejemplo positivo seria una oracion en castellano y su traduccion correcta en shiwilu. Un negativo facil seria una frase completamente distinta. Eso no ensena mucho. Un hard negative, en cambio, es una frase incorrecta pero parecida, que obliga al modelo a aprender diferencias finas.

El entrenamiento busca que el modelo haga esto:

```text
acercar la traduccion correcta
alejar traducciones parecidas pero incorrectas
```

Ademas, el ajuste fue bidireccional:

```text
castellano -> shiwilu
shiwilu -> castellano
```

Por eso el modelo final se presenta como un recurso bilingue bidireccional.

Resultados aproximados del modelo candidato:

- castellano -> shiwilu: `R@1 = 0.7882`
- shiwilu -> castellano: `R@1 = 0.7913`

Esto significa que, en casi 8 de cada 10 consultas, el modelo recupera como primera opcion la oracion correcta.

## 4. Preparacion para NMT

Luego se preparo el sistema de traduccion automatica neuronal. Los scripts principales estan en:

```text
scripts/nmt/
```

Scripts relevantes:

- `scripts/nmt/10_canonicalize_dataset.py`
- `scripts/nmt/20_semantic_filter.py`
- `scripts/nmt/21_build_faiss.py`
- `scripts/nmt/22_train_sentencepiece.py`
- `scripts/nmt/30_train_lora.py`

En esta etapa se construyo el dataset canonico para NMT, se uso el modelo de embeddings como filtro semantico y se prepararon indices FAISS para busqueda vectorial rapida.

FAISS permite buscar vectores cercanos eficientemente. En esta tesis ayuda a encontrar oraciones semanticamente similares y a controlar la calidad del corpus o de los datos aumentados.

Tambien se probo SentencePiece, pero el sistema final se apoyo en NLLB, que ya tiene tokenizacion multilingue robusta.

## 5. Modelo NMT usado

El modelo base del sistema de traduccion fue:

```text
NLLB-200 distilled
```

NLLB significa "No Language Left Behind", un modelo de Meta entrenado para traduccion multilingue. Se eligio porque shiwilu es una lengua de bajos recursos y no convenia entrenar un traductor completo desde cero.

No se ajusto todo el modelo completo. Se uso:

```text
LoRA
```

LoRA permite adaptar un modelo grande entrenando pocos parametros adicionales. Esto reduce costo computacional, memoria y riesgo de sobreajuste.

El mejor modelo fue:

```text
v2.1b LoRA+
NLLB-200 + LoRA r=32, alpha=64
lr_B = 16 * lr_A
corpus xl
```

En palabras simples:

```text
tomamos un traductor multilingue fuerte,
lo adaptamos al par shiwilu-castellano con LoRA,
y usamos una configuracion LoRA+ que entreno mejor que las variantes anteriores.
```

## 6. Aumentacion de datos

Como el corpus era pequeno, se exploraron estrategias de aumentacion:

- backtranslation;
- round-trip backtranslation;
- mineria de pares;
- variantes morfologicas revisables.

Scripts relacionados:

- `scripts/nmt/60_backtranslate.py`
- `scripts/nmt/60b_roundtrip_bt.py`
- `scripts/nmt/61_mine_pairs.py`
- `scripts/nmt/62_morph_variants.py`
- `scripts/nmt/63_train_with_augmented.py`

La backtranslation clasica desde shiwilu no fue tan util porque habia pocas oraciones monolingues shiwilu. Por eso se uso mejor una estrategia desde castellano con OPUS-100 y filtros semanticos.

La idea fue:

```text
generar pares sinteticos,
filtrarlos con embeddings,
y entrenar variantes NMT para medir si mejoraban.
```

Una conclusion importante es que mas datos no siempre producen un mejor modelo. La calidad de los datos sinteticos debe controlarse, porque los pares ruidosos pueden degradar el entrenamiento.

## 7. Evaluacion automatica

La evaluacion del NMT se hizo con:

- BLEU;
- chrF++;
- COMET;
- analisis de tokens raros;
- intervalos de confianza bootstrap;
- reranking semantico.

Scripts principales:

- `scripts/nmt/40_evaluate.py`
- `scripts/nmt/41_rare_token_eval.py`
- `scripts/nmt/50_rerank.py`
- `scripts/nmt/70_compare_runs.py`
- `scripts/nmt/72_leaderboard.py`
- `scripts/nmt/73_bootstrap_ci.py`
- `scripts/nmt/74_thesis_tables_phase6.py`

La metrica mas util para esta tesis fue chrF++, porque trabaja a nivel de caracteres y subpalabras. Eso es importante para lenguas con morfologia compleja o con poca estandarizacion ortografica. BLEU puede ser demasiado estricto si la traduccion usa formas distintas pero aceptables.

El mejor resultado fue:

```text
v2.1b LoRA+ reranked
avg chrF++ = 44.99
IC 95% = [43.17, 46.96]
```

El reranking significa que el modelo genera varias hipotesis y luego se reordena usando una senal semantica. En este caso, el embedding bilingue ayuda a elegir una traduccion mas cercana al significado esperado.

## 8. Protocolo de validacion humana

El Capitulo 6 corresponde al OE3. No afirma que ya se hizo validacion con hablantes. Lo que documenta es que se dejo preparado un protocolo para hacerla correctamente.

Script principal:

```text
scripts/nmt/71_human_eval_template.py
```

Artefactos generados:

- `reports/05_nmt/evaluation_xl/human_eval_template.csv`
- `reports/05_nmt/evaluation_xl/human_eval_anon_key.json`
- `reports/05_nmt/evaluation_xl/human_eval_protocol.md`

La muestra preparada tiene:

```text
100 items por direccion
200 items en total
```

Evalua sistemas anonimizados como A/B/C/D, sin revelar al evaluador cual sistema produjo cada hipotesis.

La rubrica usa escala 1-5 para:

- adecuacion;
- fluidez;
- pertinencia cultural.

Esto es importante porque BLEU, chrF++ y COMET no reemplazan a hablantes humanos, especialmente en una lengua originaria. Las metricas automaticas dicen si el modelo se parece a una referencia, pero no siempre capturan naturalidad, aceptabilidad cultural o variantes validas.

## 9. Como explicar la tesis en una defensa

Una forma clara de explicarlo seria:

> Mi tesis no consistio unicamente en entrenar un traductor. Primero construi un pipeline reproducible para depurar y auditar un corpus paralelo shiwilu-castellano. Luego entrene un modelo de embeddings bilingue basado en multilingual E5, ajustado con mineria controlada e iterativa de negativos dificiles, para obtener una senal semantica util. Ese modelo sirvio como recurso auxiliar para filtrar datos, analizar similitud y apoyar el reranking del sistema NMT. Despues adapte NLLB-200 con LoRA y LoRA+, porque entrenar un modelo completo desde cero no era viable para una lengua de bajos recursos. Finalmente evalue las variantes con metricas automaticas, intervalos bootstrap y analisis de tokens raros, y prepare un protocolo de validacion participativa para hablantes, sin inventar resultados humanos no ejecutados.

La logica completa es:

```text
calidad de datos
-> representacion semantica
-> adaptacion eficiente de modelo multilingue
-> evaluacion automatica reproducible
-> validacion humana preparada
```

Esa es la columna vertebral de la tesis.
