$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runBat = Join-Path $repoRoot "run_windows.bat"
$outputDir = Join-Path $repoRoot "artifacts\preprocess-image-diagnostic"

if (-not (Test-Path -LiteralPath $runBat -PathType Leaf)) {
    throw "run_windows.bat not found: $runBat"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$env:D2S_PREPROCESS_IMAGE_DIAGNOSTIC = "1"
$env:D2S_PREPROCESS_IMAGE_DIAGNOSTIC_DELAY_S = "3.0"
$env:D2S_PREPROCESS_IMAGE_DIAGNOSTIC_DIR = $outputDir

Write-Host "Pre-inference image diagnostic enabled."
Write-Host "The exact RGB frame after capture scaling and before inference will be saved once."
Write-Host "For a downscaled capture, Area, Lanczos4, Bicubic, and Area+RCAS candidates are exported from the same raw frame."
Write-Host "Output directory: $outputDir"
Write-Host "Run 1: capture at 3840x2160 and select GUI Render Scale '1K / 50%'."
Write-Host "Run 2: capture natively at 1920x1080 and run this script again."
Write-Host "Compare the two PNG files whose names include capture and render dimensions."
Write-Host "Also inspect the downsample_compare_* directory created by the 4K run."

& $runBat
exit $LASTEXITCODE
