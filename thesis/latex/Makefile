# =====================================================================
# Makefile multiplataforma para compilar la tesis LaTeX.
#
# Requisitos:
#   - GNU Make
#   - Python 3 accesible como `python3` o `python`
#   - xelatex y biber en PATH
#
# Uso:
#   make              # compila pdf/tesis.pdf
#   make latexmk      # compila usando latexmk si esta instalado
#   make clean        # elimina auxiliares/build
#   make distclean    # elimina auxiliares/build y pdf/tesis.pdf
#   make PYTHON=python # usar si Windows no expone python3
#
# En Windows se recomienda ejecutar desde Git Bash, MSYS2, WSL o similar.
# =====================================================================

MAIN := tesis
TEX := $(MAIN).tex
BIB := $(MAIN).bib
BUILD_DIR := build
PDF_DIR := pdf
PDF := $(PDF_DIR)/$(MAIN).pdf

TEXLIVE_BIN ?= $(firstword $(wildcard /Library/TeX/texbin /usr/local/texlive/*/bin/universal-darwin /usr/local/texlive/*/bin/*-darwin))

ifneq ($(TEXLIVE_BIN),)
export PATH := $(TEXLIVE_BIN):$(PATH)
endif

PYTHON ?= $(shell if command -v python3 >/dev/null 2>&1; then printf python3; elif command -v python >/dev/null 2>&1; then printf python; else printf python; fi)
XELATEX ?= $(if $(TEXLIVE_BIN),$(TEXLIVE_BIN)/xelatex,xelatex)
BIBER ?= $(if $(TEXLIVE_BIN),$(TEXLIVE_BIN)/biber,biber)
LATEXMK ?= $(if $(TEXLIVE_BIN),$(TEXLIVE_BIN)/latexmk,latexmk)

LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error
XELATEX_BUILD := $(XELATEX) $(LATEX_FLAGS) -output-directory=$(BUILD_DIR) $(TEX)

.PHONY: all pdf manual latexmk dirs check-tools check-latexmk clean distclean help

all: pdf

pdf: manual
	$(PYTHON) -c "from pathlib import Path; import shutil; src=Path('$(BUILD_DIR)')/'$(MAIN).pdf'; dst=Path('$(PDF)'); dst.parent.mkdir(exist_ok=True); shutil.copy2(src, dst); print(f'PDF generado: {dst}')"

manual: check-tools dirs $(TEX) $(BIB)
	$(XELATEX_BUILD)
	$(BIBER) --input-directory $(BUILD_DIR) --output-directory $(BUILD_DIR) $(MAIN)
	$(XELATEX_BUILD)
	$(XELATEX_BUILD)

latexmk: check-latexmk dirs $(TEX) $(BIB)
	$(LATEXMK) -xelatex $(LATEX_FLAGS) -outdir=$(BUILD_DIR) $(TEX)
	$(PYTHON) -c "from pathlib import Path; import shutil; src=Path('$(BUILD_DIR)')/'$(MAIN).pdf'; dst=Path('$(PDF)'); dst.parent.mkdir(exist_ok=True); shutil.copy2(src, dst); print(f'PDF generado: {dst}')"

dirs:
	$(PYTHON) -c "from pathlib import Path; Path('$(BUILD_DIR)').mkdir(exist_ok=True); Path('$(PDF_DIR)').mkdir(exist_ok=True)"

check-tools:
	$(PYTHON) -c "import shutil, sys; missing=[cmd for cmd in ['$(XELATEX)', '$(BIBER)'] if shutil.which(cmd) is None]; sys.exit('Faltan herramientas en PATH: ' + ', '.join(missing)) if missing else None"

check-latexmk:
	$(PYTHON) -c "import shutil, sys; sys.exit('Falta latexmk en PATH') if shutil.which('$(LATEXMK)') is None else None"

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; shutil.rmtree('$(BUILD_DIR)', ignore_errors=True); [p.unlink(missing_ok=True) for p in Path('.').glob('$(MAIN).*') if p.suffix not in {'.tex', '.bib', '.pdf'}]; [p.unlink(missing_ok=True) for p in Path('.').glob('*.aux')]"

distclean: clean
	$(PYTHON) -c "from pathlib import Path; import shutil; Path('$(PDF)').unlink(missing_ok=True); shutil.rmtree('$(PDF_DIR)', ignore_errors=True)"

help:
	@echo "Targets disponibles:"
	@echo "  make / make pdf      Compila la tesis en $(PDF)"
	@echo "  make latexmk         Compila con latexmk si esta instalado"
	@echo "  make clean           Elimina auxiliares y build/"
	@echo "  make distclean       Elimina auxiliares, build/ y pdf/"
