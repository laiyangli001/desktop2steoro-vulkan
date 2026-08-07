$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

# Keep the normal per-eye Projection path, but skip Vulkan SBS/Glow overlays
# so the headset shows only Filament environment/controllers/laser output.
$env:D2S_VULKAN_PROJECTION_COMPOSER = "1"
$env:D2S_FILAMENT_PROJECTION_ONLY = "1"
$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "0"
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "D2S_VULKAN_PROJECTION_COMPOSER=1"
Write-Host "D2S_FILAMENT_PROJECTION_ONLY=1"
Write-Host "D2S_VULKAN_PROJECTION_QUALITY_CHAIN=0"
Write-Host "Expected log: Filament projection-only diagnostic active"
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
