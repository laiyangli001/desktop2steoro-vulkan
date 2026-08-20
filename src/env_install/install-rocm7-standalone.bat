
@echo off
setlocal
echo --- Desktop2Stereo Installer (With ROCm7 for AMD GPU/APUs.) ---
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON_ROOT=%PROJECT_ROOT%\python3"
set "PYTHONPATH=%PYTHON_ROOT%;%PYTHONPATH%"
set "PYTHON_EXE=%PYTHON_ROOT%\python.exe"
set "PIP_RETRIES=10"
set "PIP_TIMEOUT=180"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Project-local Python was not found at "%PYTHON_EXE%".
    echo Install the complete Python 3.12 x64 runtime in src\python3 first.
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize.bit_length() == 63 else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: ROCm7 requires the project-local Python 3.12 x64 runtime.
    pause
    exit /b 1
)

REM --- Get AMD GPUs sorted by AdapterRAM (largest first) ---
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object Name -like '*AMD*' | Sort-Object AdapterRAM -Descending | ForEach-Object { $_.Name }"`) do (
    set "FULL_GPU_NAME=%%i"
    goto :ProcessGPU
)


echo ERROR: No AMD GPU detected.
pause
exit /b 1

:ProcessGPU
echo Detected GPU: %FULL_GPU_NAME%




:InstallDependencies
echo - Setting up the virtual environment
REM Set paths
set "VIRTUAL_ENV=%PYTHON_ROOT%"
set "REQUIREMENTS_FILE=%SCRIPT_DIR%requirements-rocm7.txt"

echo.
echo Installing requirements from: %REQUIREMENTS_FILE%
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%" --no-cache-dir --no-warn-script-location --retries 5 --timeout 120
if errorlevel 1 (
    echo Failed to install ROCm requirements
    pause
    exit /b 1
)
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --no-cache-dir --no-warn-script-location --retries 5 --timeout 120
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m rocm_sdk init
if errorlevel 1 (
    echo Failed to initialize the ROCm SDK
    pause
    exit /b 1
)
echo Python environment deployed successfully.

echo To enable torch.compile on AMD ROCm7 supported GPUs, you must install vs_buildtools https://aka.ms/vs/17/release/vs_buildtools.exe and select the "Desktop development with C++" to install (~6GB). OR you can just run with the torch.compile unchecked.

pause
exit /b 0
