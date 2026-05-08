# Como compilar la tesis

Esta guia explica como generar el PDF de la tesis en macOS y Windows. El
archivo final se genera en:

```bash
thesis/latex/pdf/tesis.pdf
```

## Requisitos

Para compilar la tesis se necesitan estas herramientas:

- **Python 3**, disponible como `python3` o `python`.
- **GNU Make**, para usar los comandos del `Makefile`.
- **XeLaTeX** y **Biber**, porque la tesis usa fuentes del sistema y bibliografia
  con `biblatex`.

Opcionalmente se puede instalar `latexmk`, pero no es obligatorio. El comando
principal del repositorio ya ejecuta la secuencia manual:

```bash
xelatex -> biber -> xelatex -> xelatex
```

## Compilar desde macOS

1. Instalar MacTeX o MacTeX no GUI:

```bash
brew install --cask mactex-no-gui
```

2. Cerrar y volver a abrir la terminal. Si `xelatex` no aparece en el `PATH`,
   normalmente se encuentra en:

```bash
/Library/TeX/texbin
```

El `Makefile` intenta detectar automaticamente esa ruta en macOS.

3. Desde la raiz del proyecto, ejecutar:

```bash
make thesis
```

Tambien se puede entrar a la carpeta LaTeX y compilar directamente:

```bash
cd thesis/latex
make
```

## Compilar desde Windows

En Windows se recomienda usar **Git Bash**, **MSYS2** o **WSL**, porque el
`Makefile` usa comandos compatibles con entorno tipo Unix.

1. Instalar una distribucion LaTeX:

- MiKTeX: https://miktex.org/download
- TeX Live: https://tug.org/texlive/windows.html

2. Instalar Python 3:

```bash
python --version
```

3. Verificar que las herramientas esten disponibles:

```bash
xelatex --version
biber --version
make --version
```

4. Desde la raiz del proyecto, ejecutar:

```bash
make thesis PYTHON=python
```

Si `python3` existe en tu entorno, tambien puedes usar simplemente:

```bash
make thesis
```

## Comandos utiles

Compilar la tesis desde la raiz del repositorio:

```bash
make thesis
```

Limpiar archivos auxiliares:

```bash
make thesis-clean
```

Limpiar archivos auxiliares y eliminar el PDF generado:

```bash
make thesis-distclean
```

Compilar desde la carpeta `thesis/latex`:

```bash
cd thesis/latex
make
```

Compilar con `latexmk`, si esta instalado:

```bash
cd thesis/latex
make latexmk
```

## Compilacion manual alternativa

Si no se quiere usar `make`, se puede compilar manualmente desde
`thesis/latex`:

```bash
mkdir -p build pdf
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build tesis.tex
biber --input-directory build --output-directory build tesis
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build tesis.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build tesis.tex
cp build/tesis.pdf pdf/tesis.pdf
```

En Windows, si `mkdir -p` o `cp` no funcionan, usa el `Makefile` desde Git Bash,
MSYS2 o WSL.

## Errores comunes

### `xelatex: command not found`

Significa que LaTeX no esta instalado o que su carpeta de binarios no esta en
el `PATH`. En macOS, revisar:

```bash
ls /Library/TeX/texbin/xelatex
```

En Windows, revisar que MiKTeX o TeX Live hayan agregado sus binarios al `PATH`.

### `biber: command not found`

La bibliografia requiere `biber`, no `bibtex`. Instala una distribucion LaTeX
completa o agrega `biber` al `PATH`.

### Error con `python3` en Windows

Algunos entornos de Windows exponen Python como `python`, no como `python3`.
Usa:

```bash
make thesis PYTHON=python
```

### El PDF no se actualiza

Ejecuta una limpieza y vuelve a compilar:

```bash
make thesis-clean
make thesis
```

## Resultado esperado

Si todo compila correctamente, deberia aparecer un mensaje similar a:

```bash
PDF generado: pdf/tesis.pdf
```

El PDF final quedara en `thesis/latex/pdf/tesis.pdf`.
