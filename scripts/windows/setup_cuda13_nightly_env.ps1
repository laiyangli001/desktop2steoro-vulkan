param(
    [string]$PythonVersion = "3.14.6",
    [string]$PythonDir = "python-cu13"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Downloads = Join-Path $Root "downloads"
$PythonTarget = Join-Path $Root $PythonDir
$Installer = Join-Path $Downloads "python-$PythonVersion-amd64.exe"
$Url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"

New-Item -ItemType Directory -Force $Downloads | Out-Null

if (-not (Test-Path $Installer)) {
    Write-Host "[1/5] Downloading Python $PythonVersion ..."
    Invoke-WebRequest -Uri $Url -OutFile $Installer
} else {
    Write-Host "[1/5] Python installer already exists: $Installer"
}

if (-not (Test-Path (Join-Path $PythonTarget "python.exe"))) {
    Write-Host "[2/5] Installing isolated Python to $PythonTarget ..."
    New-Item -ItemType Directory -Force $PythonTarget | Out-Null
    Start-Process -FilePath $Installer -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=$PythonTarget",
        "Include_launcher=0",
        "PrependPath=0",
        "Include_test=0",
        "Include_pip=1"
    ) -Wait -NoNewWindow
} else {
    Write-Host "[2/5] Isolated Python already exists: $PythonTarget"
}

$PythonExe = Join-Path $PythonTarget "python.exe"
& $PythonExe --version

Write-Host "[3/4] Installing PyTorch nightly cu130 ..."
& $PythonExe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

Write-Host "[4/4] Installing latest experiment extras ..."
& $PythonExe -m pip install --pre --upgrade -r (Join-Path $Root "requirements-cuda13-nightly-extra.txt")

Write-Host "[done] CUDA 13 nightly portable runtime ready: $PythonTarget"
& $PythonExe -B -c "import torch, onnxruntime as ort; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('ort', ort.__version__, ort.get_available_providers())"
