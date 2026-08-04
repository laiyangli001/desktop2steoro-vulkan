$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

# Enable the Filament per-eye shader diagnostic for this run only.
$env:D2S_FILAMENT_EYE_DIAGNOSTIC = "1"

Write-Host "D2S_FILAMENT_EYE_DIAGNOSTIC=1"
Write-Host "Expected result: left eye red, right eye green."
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
