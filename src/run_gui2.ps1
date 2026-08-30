[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$srcRoot = $PSScriptRoot
$appDir = Join-Path $srcRoot "desktop2stereo"
$pythonExe = Join-Path $srcRoot "python3\python.exe"
$gui2Entry = Join-Path $appDir "gui2\gui.py"
$mainEntry = Join-Path $appDir "main.py"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Desktop2Stereo Python was not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $gui2Entry -PathType Leaf)) {
    throw "Desktop2Stereo GUI2 entry was not found: $gui2Entry"
}

if (-not (Test-Path -LiteralPath $mainEntry -PathType Leaf)) {
    throw "Desktop2Stereo application entry was not found: $mainEntry"
}

$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$previousPythonExe = [Environment]::GetEnvironmentVariable("PYTHON_EXE", "Process")
$exitCode = 1

try {
    $env:PYTHONPATH = $appDir
    $env:PYTHON_EXE = $pythonExe

    Write-Host "Starting Desktop2Stereo GUI2..."
    Write-Host "Application directory: $appDir"

    Push-Location -LiteralPath $appDir
    try {
        & $pythonExe $mainEntry --gui2
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }

    if ($null -eq $previousPythonExe) {
        Remove-Item Env:PYTHON_EXE -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHON_EXE = $previousPythonExe
    }
}

exit $exitCode
