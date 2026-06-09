param(
    [switch]$SinEnlace,
    [switch]$SinRerank,
    [int]$Puerto = 7860
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv-nmt\Scripts\python.exe"
$App = Join-Path $Root "app.py"

if (-not (Test-Path $Python)) {
    throw "No se encontro el Python de .venv-nmt en: $Python"
}

if (-not (Test-Path $App)) {
    throw "No se encontro app.py en: $App"
}

$ArgsList = @($App, "--port", $Puerto)
if ($SinEnlace) { $ArgsList += "--no-share" }
if ($SinRerank) { $ArgsList += "--no-rerank" }

Write-Host "Lanzando interfaz web del traductor castellano <-> shiwilu..."
if ($SinEnlace) {
    Write-Host "Modo: solo local (127.0.0.1:$Puerto)"
} else {
    Write-Host "Modo: local + enlace publico temporal (*.gradio.live)"
}
Write-Host ""

& $Python @ArgsList
