$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

# Disable the experimental Filament external VkImage zero-copy path for this run only.
$env:D2S_ENABLE_FILAMENT_SCREEN_IMAGE = "0"

Write-Host "D2S_ENABLE_FILAMENT_SCREEN_IMAGE=0"
Write-Host "Starting the normal Windows launcher for the fallback A/B test..."

& $runBat
exit $LASTEXITCODE
