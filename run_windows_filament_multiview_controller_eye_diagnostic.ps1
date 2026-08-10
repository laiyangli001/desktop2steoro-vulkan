$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$baseDiagnostic = Join-Path $repoRoot "run_windows_filament_multiview_projection_diagnostic.ps1"

if (-not (Test-Path -LiteralPath $baseDiagnostic -PathType Leaf)) {
    throw "Filament multiview diagnostic not found: $baseDiagnostic"
}

$env:D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC = "1"
$env:D2S_FILAMENT_EYE_DIAGNOSTIC = "1"
$env:D2S_FILAMENT_MULTIVIEW_LAYER_READBACK = "1"

Write-Host "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC=1"
Write-Host "D2S_FILAMENT_EYE_DIAGNOSTIC=1 (backend stereo trace enabled)"
Write-Host "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK=1 (one-shot layer 0/1 GPU readback)"
Write-Host "Controller geometry replaces its normal materials; fragment getEyeIndex() selects:"
Write-Host "  left eye controller: solid red"
Write-Host "  right eye controller: solid green"
Write-Host "Environment remains unchanged; SBS and Glow remain disabled."
Write-Host "After exit, preserve every '[D2S stereo trace]' log line, including fragmentViewIndex."
Write-Host "Also preserve the '[OpenXRViewer] Filament multiview layer readback' line and PNG."

& $baseDiagnostic
exit $LASTEXITCODE
