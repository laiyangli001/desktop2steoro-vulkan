$ErrorActionPreference = "Stop"

$runBat = Join-Path $PSScriptRoot "run_windows.bat"
if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_VULKAN_PROJECTION_COMPOSER = "1"
$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "0"
$env:D2S_FILAMENT_PROJECTION_ONLY = "0"
$env:D2S_FILAMENT_DEPTH_SWAPCHAIN = "0"
$env:D2S_FILAMENT_CONTROLLER_OVERLAY_AFTER_COMPOSER = "1"
$env:D2S_OPENXR_DEBUG = "1"
Remove-Item Env:D2S_OPENXR_SCREEN_QUAD_REPROJECTION -ErrorAction SilentlyContinue
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue

Write-Host "Vulkan Projection Composer LOD0 isolation"
Write-Host "Filament environment/controllers + basic SBS overlay"
Write-Host "Controller/laser foreground is rendered again after screen and Glow"
Write-Host "Quality chain, depth experiment, and projection-only mode are disabled"
Write-Host "Expected log: Vulkan projection composer active: mode=graphics_triangle_strip"
Write-Host "Expected log: Filament controller overlay active"
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
