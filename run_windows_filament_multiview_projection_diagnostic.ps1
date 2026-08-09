$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC = "1"
$env:D2S_VULKAN_PROJECTION_COMPOSER = "0"
$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "0"
Remove-Item Env:D2S_FILAMENT_PROJECTION_ONLY -ErrorAction SilentlyContinue
Remove-Item Env:D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "Minimal Filament multiview Projection test"
Write-Host "  Projection swapchain: one array_size=2 target"
Write-Host "  Filament submission: one stereoscopic render per XR frame"
Write-Host "  Vulkan SBS/Glow Composer: disabled"
Write-Host "Expected result: environment/controllers have correct stereo in both eyes."
Write-Host "Close any existing Desktop2Stereo process before starting this test."
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
