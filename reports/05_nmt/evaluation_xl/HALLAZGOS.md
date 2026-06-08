# Hallazgos — Enhancements a partir de la consulta al experto en evaluación de MT

> Documento interno de consolidación. Resume, en un solo lugar, **qué nos
> aconsejó el experto en evaluación de traducción automática**, **qué hicimos**
> con ese consejo y **qué descubrimos** al aplicarlo sobre el modelo campeón.
> No es el acta de validación ni la respuesta al experto: es la base ordenada
> para armar ese paquete cuando se decida.
>
> Modelo analizado: `nllb_bidi_lora_v2_1b_loraplus_xl` (campeón, LoRA+),
> variante `xl`, conjunto de prueba (446 oraciones por dirección), salidas
> reranked.

## 1. Qué pidió el experto (y qué quería decir de fondo)

El experto revisó el proyecto sin haber leído aún el documento completo y envió
consejos de método. Reinterpretados en su intención de fondo:

1. **No mirar una sola métrica ni un solo número global.** Las dos direcciones
   no son comparables: al traducir **hacia el castellano** (shw→spa) BLEU es
   informativo; al traducir **hacia el shiwilu** (spa→shw) BLEU subestima la
   calidad porque la lengua es aglutinante y penaliza cada palabra mal formada
   aunque el sentido sea correcto. Ahí manda chrF++.
2. **Distinguir cuándo un número es ruido.** Como referencia práctica (ojo de
   experto, no regla formal), un BLEU por debajo de ~10 o un chrF++ por debajo
   de ~20 suele indicar salidas dominadas por palabras frecuentes y cortas,
   cercanas al ruido.
3. **Demostrar que el modelo traduce, no solo reportar agregados.** Pidió un
   análisis cualitativo que abra las traducciones, las separe por calidad y
   muestre con ejemplos dónde acierta y dónde falla.
4. **No inventar un ganador con decimales cuando dos modelos empatan.** Propuso
   una comparación ciega A/B: mostrar la referencia y dos traducciones sin
   revelar el sistema, y elegir cuál se acerca más a la referencia.

**En una frase:** no quedarse en la tabla de métricas; demostrar con evidencia
cualitativa que el sistema realmente traduce, y ser honesto cuando los números
no alcanzan para decidir.

## 2. Qué hicimos (mapa consejo → entrega)

| Consejo | Entrega | Ubicación |
|---|---|---|
| Métrica por dirección (BLEU shw→spa, chrF++ spa→shw); chrF++ sigue siendo el criterio global | Lectura por dirección añadida a la redacción | tesis §2 (chrF), §5 (evaluación), §6 |
| Umbral de ruido como heurística interpretativa | Párrafo de interpretación v0 vs. campeón | tesis §5 (`nmt-resultados`) |
| Análisis cualitativo estratificado | Nueva subsección + tabla + script | tesis §5.2.7 (`nmt-cualitativo`), `42_qualitative_analysis.py` |
| Justificar la selección ante empate estadístico | Párrafo de parsimonia (v2.1b sobre v2.1) | tesis §5 |
| Preferencia pareada ciega | Nueva subsección + instrumento Excel ciego | tesis §6.2.1 (`protocolo-preferencia-pareada`), `76_pairwise_preference.py` |

Ya estaba implementado de antes (el experto lo recomendaba y no hubo que
tocarlo): chrF++ como métrica primaria, reporte de ambas direcciones,
intervalos de confianza bootstrap, honestidad sobre el empate v2.1≈v2.1b y el
protocolo de evaluación participativa preparado.

## 3. Qué descubrimos al aplicarlo

### 3.1 El sistema traduce de verdad: no es ruido

Distribución de las salidas del campeón por banda de puntaje por oración:

**shw→spa (BLEU por oración, n=446, media 40.40)**

| Banda | n | % | media |
|---|---:|---:|---:|
| BLEU < 10 (ruido) | 77 | 17.3 % | 5.94 |
| BLEU 10–20 | 57 | 12.8 % | 14.21 |
| BLEU ≥ 20 | 312 | 70.0 % | 53.69 |

**spa→shw (chrF++ por oración, n=446, media 49.07)**

| Banda | n | % | media |
|---|---:|---:|---:|
| chrF++ < 20 (ruido) | 51 | 11.4 % | 13.55 |
| chrF++ 20–40 | 149 | 33.4 % | 30.95 |
| chrF++ ≥ 40 | 246 | 55.2 % | 67.41 |

→ El 70 % de las salidas hacia el castellano y el 55 % de las salidas hacia el
shiwilu caen en la banda alta. Solo el 17 % / 11 % queda en la banda de ruido.
La salida es mayoritariamente traducción informativa, no ruido.

### 3.2 El modelo base (v0) sí estaba en el ruido; el campeón lo deja atrás

Se confirma el umbral del experto: v0 daba BLEU 9.73 (shw→spa) y 4.61
(spa→shw), es decir, en el umbral de ruido o por debajo. El campeón lo supera
con holgura: BLEU 24.48 en shw→spa y chrF++ de 42 a 48 en ambas direcciones
(muy por encima de 20). **El salto no es cosmético: separa un sistema apenas
distinguible del ruido de uno que produce traducción con contenido.**

### 3.3 Los errores de la banda baja son pérdidas reales de sentido

Esto valida usar las métricas: un BLEU bajo no penaliza paráfrasis válidas,
señala errores genuinos. El sistema suele conservar el marco sintáctico pero
yerra el contenido. Ejemplos reales:

- `a'ñapalek sukta-shunka' iskun ekkilala` → *tengo doscientos metros de altura*
  (referencia: *tengo sesentinueve años*). La estructura *tengo + cantidad* es
  correcta, aunque el valor numérico no se preserva.
- `iyatulek' wayupi mer'cha'su'` (*me gusta la fruta madura*) → *detesto beber
  limpio* (error de polaridad y de léxico).

### 3.4 La asimetría entre direcciones, ahora explicada

Hacia el shiwilu el modelo acierta los lexemas pero a veces no completa la forma
plena de la palabra (morfología aglutinante). Por eso chrF++ es la métrica justa
en esa dirección: captura las coincidencias parciales de caracteres que BLEU
descarta. La banda baja de spa→shw se concentra, de forma previsible, en
oraciones largas y con alta proporción de términos fuera de vocabulario,
coherente con la baja recuperación OOV ya documentada.

### 3.5 El empate v2.1 ≈ v2.1b es real y honesto

v2.1 (DoRA+LoRA+) y v2.1b (LoRA+ solo) son estadísticamente indistinguibles
(intervalos de confianza traslapados). La selección se resuelve por parsimonia:
v2.1b alcanza el mismo nivel con un adaptador más simple. El instrumento de
preferencia pareada es exactamente la herramienta para desempatar estos casos
sin recurrir a más decimales. Métrica global del campeón: avg chrF++ = 44.99
(IC 95 % [43.17, 46.96]).

## 4. Conclusión

Al seguir el consejo del experto pasamos de *"el modelo da estos números"* a
*"el modelo traduce con sentido; esto es lo que hace bien, esto lo que le
cuesta, y aquí está la evidencia"*. Es precisamente el tipo de evidencia que un
evaluador necesita para validar el trabajo.

## 5. Pendiente (a propósito)

- **Evaluación participativa con hablantes**: preparada (rúbrica + preferencia
  pareada) pero **no ejecutada**. Declarada como trabajo futuro en el Cap. 6.
- **Acta de validación del experto** (documento firmable): no iniciada; es el
  último paso.
- **Respuesta al experto**: no redactada hasta cerrar todo lo anterior.

## 6. Reproducir los artefactos

```powershell
# Análisis cualitativo estratificado
.venv-nmt/Scripts/python -m scripts.nmt.42_qualitative_analysis --variant xl

# Instrumento de preferencia pareada ciega (v2.1b vs v2.1)
.venv-nmt/Scripts/python -m scripts.nmt.76_pairwise_preference --variant xl
```

Salidas asociadas:

- `nllb_bidi_lora_v2_1b_loraplus_xl/qualitative/{bucket_summary.json, sampled_examples.csv, qualitative_report.md}`
- `pairwise_preference.xlsx` + `pairwise_preference_anon_key.json` (clave anónima: mantener privada)
