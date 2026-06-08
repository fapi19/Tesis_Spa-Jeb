# Análisis cualitativo estratificado — nllb_bidi_lora_v2_1b_loraplus_xl

- Predicciones: reranked
- Muestras por bucket: 5 (seed 2026)

shw→spa se puntúa con BLEU por oración; spa→shw con chrF++ por oración. Cortes según el umbral de ruido del experto (BLEU≤10 / chrF++≤20 ≈ ruido).

## shw2spa (métrica: bleu, n=446, media=40.40)

| Bucket | n | % | media |
|---|---:|---:|---:|
| bleu_0_10 | 77 | 17.3% | 5.94 |
| bleu_10_20 | 57 | 12.8% | 14.21 |
| bleu_20_plus | 312 | 70.0% | 53.69 |

### Ejemplos — bleu_0_10
- **score=0.0** (`flashcards2`)
  - fuente: imanan
  - referencia: susto.
  - hipótesis: panza
- **score=8.13** (`pdf_textos`)
  - fuente: nanapa yunsu'i kenmu'wa'kinsa' ñii asu' samer,
  - referencia: para que salgan para nosotros nomás, para que sean para nosotros nomás estos peces,
  - hipótesis: por eso saliendo nosotros somos estos peces,
- **score=9.65** (`fidel_lomas`)
  - fuente: a'ñapalek sukta-shunka' iskun ekkilala.
  - referencia: tengo sesentinueve años.
  - hipótesis: tengo doscientos metros de altura.
- **score=9.65** (`pdf_textos`)
  - fuente: kenmu'wei'na ipa' ñinchillintekwa'.
  - referencia: ahora nosotros conocemos los nombres.
  - hipótesis: nosotros ya lo hemos aprendido.
- **score=9.69** (`flashcards_oraciones`)
  - fuente: iyatulek' wayupi mer'cha'su'
  - referencia: me gusta la fruta madura.
  - hipótesis: detesto beber limpio.

### Ejemplos — bleu_10_20
- **score=10.55** (`fidel_lomas`)
  - fuente: yalli'lusa' pa'ter'kasu'nta' chimiñina',
  - referencia: los señores importantes murieron,
  - hipótesis: los hermanos que iban a ir también murieron,
- **score=12.44** (`flashcards_oraciones`)
  - fuente: ipa'la nakusu' ukawañi'
  - referencia: hoy hace mucho calor.
  - hipótesis: ahora está frío.
- **score=13.22** (`pdf_textos`)
  - fuente: kui'na kaluwi'mu innichi'nek ñi enñupa' pa'a'kasu'.
  - referencia: en cambio yo por estar enferma no puedo ir a ninguna parte.
  - hipótesis: pero yo con mi enfermedad no puedo ir donde quiera.
- **score=17.54** (`pdf_textos`)
  - fuente: nu'sik upetchununta'lek deklek.
  - referencia: luego se vuelve a agregar agua.
  - hipótesis: después se vuelve a levantar el agua con el agua.
- **score=19.0** (`flashcards2`)
  - fuente: kenma chiminllinkekla nampila
  - referencia: tú sobreviviste.
  - hipótesis: tú desapareces temprano.

### Ejemplos — bleu_20_plus
- **score=50.0** (`flashcards2`)
  - fuente: apu'ker'
  - referencia: aflójala.
  - hipótesis: párale.
- **score=55.03** (`flashcards2`)
  - fuente: linsercher' asek
  - referencia: firme aquí.
  - hipótesis: anota aquí.
- **score=60.65** (`flashcards2`)
  - fuente: wa'tenker'la
  - referencia: sólo espera.
  - hipótesis: espera.
- **score=100.0** (`flashcards2`)
  - fuente: ¡tekkinchi nuka'a!
  - referencia: ¡es verdad!
  - hipótesis: ¡es verdad!
- **score=100.0** (`flashcards_oraciones`)
  - fuente: ¿eñupi'na pidek'pen?
  - referencia: ¿dónde está tu casa?
  - hipótesis: ¿dónde está tu casa?

## spa2shw (métrica: chrf_pp, n=446, media=49.07)

| Bucket | n | % | media |
|---|---:|---:|---:|
| chrf_0_20 | 51 | 11.4% | 13.55 |
| chrf_20_40 | 149 | 33.4% | 30.95 |
| chrf_40_plus | 246 | 55.2% | 67.41 |

### Ejemplos — chrf_0_20
- **score=8.85** (`flashcards2`)
  - fuente: se desmayó.
  - referencia: muisu' kankatulli
  - hipótesis: chi'yek'lli
- **score=12.36** (`flashcards2`)
  - fuente: no sonrías.
  - referencia: yayamerladata
  - hipótesis: yawellektama'
- **score=16.02** (`pdf_textos`)
  - fuente: ¡ahí está, la candela agarren!
  - referencia: - ¡ma'ata'na pen maku'!
  - hipótesis: ¡nanek ñapalli, pen pilli'ter'!
- **score=16.19** (`pdf_textos`)
  - fuente: ahora les voy a contar cómo quiza cuando vivían los antiguos indios shawala pelearon contra los que vinieron de otro pueblo, los catellano- hablantes.
  - referencia: ipa'la' winterkenma' ma'pu'si'pa' napi' shawala kenma' ñapanna'pi'la idenmalli'na' asu' nerñinalukla uklusa'lek, kaschilla lunlusa'.
  - hipótesis: ipa'la wintetchenma' ma'pu'si'pa' nanpi'pa' nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana nana …
- **score=17.95** (`pdf_textos`)
  - fuente: nosotros abriendo trocha en el monte estamos cazando,
  - referencia: - sekllintamudek asu' tanak yunek'amudekpu'su',
  - hipótesis: kenmu'wa' ek'ker'apallidek mutupik pa'apallidek,

### Ejemplos — chrf_20_40
- **score=20.82** (`flashcards2`)
  - fuente: ¡oh, no! (masculino)
  - referencia: ¡shaw!
  - hipótesis: ¡oh, ma'sha!
- **score=25.55** (`pdf_textos`)
  - fuente: después se escondieron.
  - referencia: nu'anna' insekkitullina'.
  - hipótesis: nanekla indaperllina'.
- **score=28.36** (`fidel_lomas`)
  - fuente: la sachavaca es un lindo animal.
  - referencia: panwali'na u'chimu añimar nuka'a.
  - hipótesis: panwala nuka'a ala'sa' musenpi.
- **score=35.52** (`flashcards2`)
  - fuente: deberíamos ayudar.
  - referencia: katu'pa'tuwinansu'
  - hipótesis: katu'pa'a'kawa'su' ñapalli
- **score=39.66** (`pdf_textos`)
  - fuente: ¿cuál quieres que sea tu santo? les preguntó.
  - referencia: ¿enkasu' a'cha santupen luwantula ña'su'? itullima.
  - hipótesis: ¿ma'nen luwantula santupen? itudeklli.

### Ejemplos — chrf_40_plus
- **score=44.12** (`flashcards2`)
  - fuente: estoy preparado.
  - referencia: ñapalek insekdipersu'
  - hipótesis: insek'diperpi ñapalek
- **score=70.06** (`flashcards2`)
  - fuente: déjame ir.
  - referencia: ta'itula'u pa'ak
  - hipótesis: ta'itula'u pa'i
- **score=85.1** (`flashcards_oraciones`)
  - fuente: ellas leen una revista.
  - referencia: nawa' luntullina' ala'sa' kerka'
  - hipótesis: nawa' luntullina' alasa' kerka'
- **score=100.0** (`flashcards2`)
  - fuente: tom hablará.
  - referencia: tom lun'echu
  - hipótesis: tom lun'echu
- **score=100.0** (`flashcards_oraciones`)
  - fuente: nosotros no trabajamos en la casa.
  - referencia: kuda ku'la saka'tapi'ñidek pidek'kek
  - hipótesis: kuda ku'la saka'tapi'ñidek pidek'kek
