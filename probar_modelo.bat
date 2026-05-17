@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%~dp0.venv-nmt\Scripts\python.exe"
set "CHECKPOINT=models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl"

if not exist "%PYTHON%" (
    echo No se encontro el Python de .venv-nmt en:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%CHECKPOINT%" (
    echo No se encontro el checkpoint esperado en:
    echo %CHECKPOINT%
    pause
    exit /b 1
)

echo Cargando modelo NMT: %CHECKPOINT%
echo Modo: mejor calidad, con reranking
echo Usa "spa: texto" para castellano -^> shiwilu, "shw: texto" para shiwilu -^> castellano, y "/quit" para salir.
echo.

"%PYTHON%" -m scripts.translate_interactive --checkpoint "%CHECKPOINT%" --rerank

pause
