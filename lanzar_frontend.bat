@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=%~dp0.venv-nmt\Scripts\python.exe"
set "APP=%~dp0app.py"

if not exist "%PYTHON%" (
    echo No se encontro el Python de .venv-nmt en:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo No se encontro app.py en:
    echo %APP%
    pause
    exit /b 1
)

echo Lanzando interfaz web del traductor castellano ^<-^> shiwilu...
echo Modo: local + enlace publico temporal (*.gradio.live)
echo Para solo local, ejecuta:  lanzar_frontend.bat --no-share
echo.

"%PYTHON%" "%APP%" %*

pause
