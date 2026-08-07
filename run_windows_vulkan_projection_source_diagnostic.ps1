$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

$env:D2S_VULKAN_PROJECTION_COMPOSER = "1"
$env:D2S_VULKAN_PROJECTION_QUALITY_CHAIN = "0"
$env:D2S_OPENXR_DEBUG = "1"
Remove-Item Env:D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC -ErrorAction SilentlyContinue
Remove-Item Env:D2S_OPENXR_VISUAL_REGRESSION_DIR -ErrorAction SilentlyContinue
Remove-Item Env:D2S_VULKAN_PROJECTION_SOURCE_SYNC_DIAGNOSTIC -ErrorAction SilentlyContinue

Write-Host "D2S_VULKAN_PROJECTION_COMPOSER=1"
Write-Host "D2S_VULKAN_PROJECTION_QUALITY_CHAIN=0"
Write-Host "D2S_OPENXR_DEBUG=1"
Write-Host "Direct source diagnostic: quality, RCAS, and MIP passes are bypassed."
Write-Host "Starting the normal Windows launcher..."

& $runBat
exit $LASTEXITCODE
