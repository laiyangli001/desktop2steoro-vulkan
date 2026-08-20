$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_VULKAN_PROJECTION_COMPOSER = "1"
$env:D2S_FILAMENT_DEPTH_SWAPCHAIN = "1"
Remove-Item Env:D2S_FILAMENT_PROJECTION_ONLY -ErrorAction SilentlyContinue
$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "0"
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "D2S_VULKAN_PROJECTION_COMPOSER=1"
Write-Host "D2S_FILAMENT_DEPTH_SWAPCHAIN=1"
Write-Host "Current expected log: depth capability create=1 query=1 output=1 requested=1"
Write-Host "Depth binding is enabled for hardware validation"
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
