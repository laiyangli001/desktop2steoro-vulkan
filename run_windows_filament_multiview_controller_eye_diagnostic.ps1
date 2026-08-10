$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$baseDiagnostic = Join-Path $repoRoot "run_windows_filament_multiview_projection_diagnostic.ps1"

if (-not (Test-Path -LiteralPath $baseDiagnostic -PathType Leaf)) {
    throw "Filament multiview diagnostic not found: $baseDiagnostic"
}

$env:D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC = "1"
$env:D2S_FILAMENT_EYE_DIAGNOSTIC = "1"
$env:D2S_FILAMENT_MULTIVIEW_LAYER_READBACK = "1"
$shaderDumpDir = Join-Path $repoRoot "artifacts"
New-Item -ItemType Directory -Force -Path $shaderDumpDir | Out-Null
$env:D2S_FILAMENT_SHADER_DUMP_DIR = $shaderDumpDir

Write-Host "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC=1"
Write-Host "D2S_FILAMENT_EYE_DIAGNOSTIC=1 (backend stereo trace enabled)"
Write-Host "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK=1 (layer 0/1 GPU readback after 30 rendered frames)"
Write-Host "D2S_FILAMENT_SHADER_DUMP_DIR=$shaderDumpDir"
Write-Host "Controller material passes vertex getEyeIndex() to the fragment stage:"
Write-Host "  eye index 0: solid-red controller"
Write-Host "  eye index 1: solid-green controller"
Write-Host "Environment remains unchanged; SBS and Glow remain disabled."
Write-Host "After exit, preserve every '[D2S stereo trace]' log line, including stereo camera data."
Write-Host "After exit, preserve artifacts/filament_controller_eye_diag.vert.spv."
Write-Host "Also preserve the '[OpenXRViewer] Filament multiview layer readback' line and PNG."

& $baseDiagnostic
exit $LASTEXITCODE
