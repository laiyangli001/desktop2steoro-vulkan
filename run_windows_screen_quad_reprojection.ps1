$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_VDXR_VULKAN_NATIVE_DIAGNOSTIC = "on"
$env:D2S_OPENXR_SCREEN_QUAD_REPROJECTION = "1"
$env:D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC = "1"
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER -ErrorAction SilentlyContinue
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "D2S_OPENXR_SCREEN_QUAD_REPROJECTION=1"
Write-Host "D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC=1"
Write-Host "D2S_OPENXR_DEBUG=1"
Write-Host "Expected headset result: left eye red, right eye green."
Write-Host "If both eyes are green, VDXR ignores per-eye Quad visibility."
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
