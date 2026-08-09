$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$baseDiagnostic = Join-Path $repoRoot "run_windows_filament_multiview_projection_diagnostic.ps1"

if (-not (Test-Path -LiteralPath $baseDiagnostic -PathType Leaf)) {
    throw "Filament multiview diagnostic not found: $baseDiagnostic"
}

$env:D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC = "1"

Write-Host "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC=1"
Write-Host "Controller geometry replaces its normal materials per view:"
Write-Host "  left eye controller: solid red"
Write-Host "  right eye controller: solid green"
Write-Host "Environment remains unchanged; SBS and Glow remain disabled."

& $baseDiagnostic
exit $LASTEXITCODE
