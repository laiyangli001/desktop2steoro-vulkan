$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_OPENXR_SCREEN_QUAD_REPROJECTION = "1"
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER -ErrorAction SilentlyContinue
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "D2S_OPENXR_SCREEN_QUAD_REPROJECTION=1"
Write-Host "D2S_OPENXR_DEBUG=1"
Write-Host "Expected log: Screen Quad Reprojection active source=<width>x<height>"
Write-Host "Expected FPS path: screen_quad_active=1 xr_path=screen_quad_reprojection"
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
