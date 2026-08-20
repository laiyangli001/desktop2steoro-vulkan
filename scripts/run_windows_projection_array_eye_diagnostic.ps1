$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$layoutFile = Join-Path $repoRoot "project_paths.env"
$layout = @{}
Get-Content -LiteralPath $layoutFile -Encoding UTF8 | ForEach-Object {
    if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*?)\s*$') {
        $layout[$matches[1].Trim()] = $matches[2].Trim()
    }
}
$srcRoot = Join-Path $repoRoot $layout["D2S_APP_DIR"]
$pythonExe = Join-Path $repoRoot ($layout["D2S_PYTHON_DIR"] + "\python.exe")

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Bundled Python not found: $pythonExe"
}

$env:PYTHONPATH = $srcRoot
$env:D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC = "1"
$env:D2S_OPENXR_DEBUG = "1"

Write-Host "Minimal VDXR Projection array test"
Write-Host "  swapchains: 1"
Write-Host "  array_size: 2"
Write-Host "  left Projection view:  imageArrayIndex=0, solid red"
Write-Host "  right Projection view: imageArrayIndex=1, solid green"
Write-Host "No capture, inference, Filament, screen, Glow, or controller is loaded."
Write-Host "Expected headset result: left eye red, right eye green."
Write-Host "Press Ctrl+C to stop before the 60-second timeout."

Push-Location $repoRoot
try {
    & $pythonExe -m tools.openxr_vulkan_smoke --seconds 60 --render-scale 0.25
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
