# Cierre Del Preprocesamiento De Embeddings

Estado: `pass`

## Resumen Por Split

### train
- Filas: 2563
- Grupos: 2387
- Filas con audit flags: 1032
- Fuentes: {'flashcards': 1791, 'pdf_textos': 772}
- Longitud Shiwlu: {'min_chars': 4, 'max_chars': 169, 'mean_chars': 25.66, 'median_chars': 17, 'max_tokens': 24}
- Longitud español: {'min_chars': 3, 'max_chars': 261, 'mean_chars': 25.65, 'median_chars': 14, 'max_tokens': 47}

### valid
- Filas: 320
- Grupos: 295
- Filas con audit flags: 127
- Fuentes: {'flashcards': 236, 'pdf_textos': 84}
- Longitud Shiwlu: {'min_chars': 5, 'max_chars': 205, 'mean_chars': 24.19, 'median_chars': 17.0, 'max_tokens': 23}
- Longitud español: {'min_chars': 5, 'max_chars': 113, 'mean_chars': 22.82, 'median_chars': 14.0, 'max_tokens': 19}

### test
- Filas: 321
- Grupos: 300
- Filas con audit flags: 133
- Fuentes: {'flashcards': 221, 'pdf_textos': 100}
- Longitud Shiwlu: {'min_chars': 5, 'max_chars': 134, 'mean_chars': 25.71, 'median_chars': 18, 'max_tokens': 14}
- Longitud español: {'min_chars': 4, 'max_chars': 210, 'mean_chars': 26.21, 'median_chars': 14, 'max_tokens': 37}

## Exclusiones

- Duplicados exactos: 3
- Texto vacío: 0

## Suffix-Aware

Estado: `experimental_no_usar_como_default`
Mantener como variante experimental; no bloquea el preprocesamiento canónico.

## Decisión

El preprocesamiento canónico queda cerrado si el estado es `pass`. La siguiente fase es entrenamiento/evaluación de embeddings, no más preprocesamiento.
