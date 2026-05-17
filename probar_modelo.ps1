param(
    [switch]$SinRerank
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv-nmt\Scripts\python.exe"
$Checkpoint = "models/nmt/nllb_bidi_lora_v2_1b_loraplus_xl"

if (-not (Test-Path $Python)) {
    throw "No se encontro el Python de .venv-nmt en: $Python"
}

if (-not (Test-Path $Checkpoint)) {
    throw "No se encontro el checkpoint esperado en: $Checkpoint"
}

$ArgsList = @(
    "-m", "scripts.translate_interactive",
    "--checkpoint", $Checkpoint
)

if (-not $SinRerank) {
    $ArgsList += "--rerank"
}

Write-Host "Cargando modelo NMT: $Checkpoint"
if ($SinRerank) {
    Write-Host "Modo: rapido, sin reranking"
} else {
    Write-Host "Modo: mejor calidad, con reranking"
}
Write-Host "Usa 'spa: texto' para castellano -> shiwilu, 'shw: texto' para shiwilu -> castellano, y '/quit' para salir."
Write-Host ""

& $Python @ArgsList
