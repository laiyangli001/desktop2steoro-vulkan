$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"
if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_OPENXR_VULKAN_MULTIVIEW_EYE_DIAGNOSTIC = "1"
$env:D2S_VULKAN_PROJECTION_COMPOSER = "0"
$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "0"
Remove-Item Env:D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
Remove-Item Env:D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC -ErrorAction SilentlyContinue
Remove-Item Env:D2S_FILAMENT_MULTIVIEW_LAYER_READBACK -ErrorAction SilentlyContinue
Remove-Item Env:D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue

Write-Host "Minimal Vulkan multiview vertex gl_ViewIndex diagnostic"
Write-Host "  Target: one OpenXR array_size=2 Projection swapchain"
Write-Host "  Draw: vertex gl_ViewIndex -> flat varying -> fragment, viewMask=0x3"
Write-Host "  Expected: left eye red, right eye green"
Write-Host "  Filament, SBS, Glow and model loading are bypassed"

& $runBat
exit $LASTEXITCODE
