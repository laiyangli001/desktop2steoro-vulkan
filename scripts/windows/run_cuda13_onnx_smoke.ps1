param(
    [string]$Rgb = "outputs\demo\fast_half_sbs.png",
    [string]$OutDir = "outputs\onnx_distill_cu13_smoke"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $Root "python-cu13\python.exe"

if (-not (Test-Path $Python)) {
    throw "CUDA 13 nightly runtime not found: $Python. Run scripts\windows\setup_cuda13_nightly_env.ps1 first."
}

Push-Location $Root
try {
    & $Python -B scripts\tools\test_distill_base_onnx.py --rgb $Rgb --device cuda --out-dir $OutDir
} finally {
    Pop-Location
}
