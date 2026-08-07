$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "1"
$env:D2S_VULKAN_PROJECTION_COMPOSER = "1"
$env:D2S_FILAMENT_DEPTH_SWAPCHAIN = "1"
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "D2S_VULKAN_PROJECTION_COMPOSER=1"
Write-Host "D2S_FILAMENT_DEPTH_SWAPCHAIN=1"
Write-Host "D2S_OPENXR_DEBUG=1"
Write-Host "Expected log: Vulkan projection composer active: mode=graphics_triangle_strip"
Write-Host "Expected log: Filament depth capability: create=1 query=1 output=1 requested=1"
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
