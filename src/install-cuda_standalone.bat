@echo off
echo --- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---
echo - Setting up the virtual environment

@REM Install a project-local Python 3.12 x64 runtime first.
@REM The full python.org EXE cannot install a second copy of the same version:
@REM it enters maintenance mode for the registered installation and ignores TargetDir.
@REM The official NuGet package provides a complete relocatable runtime layout.
Set "PYTHON_VERSION=3.12.10"
Set "SCRIPT_DIR=%~dp0"
Set "PYTHON_ROOT=%SCRIPT_DIR%python3"
Set "PYTHON_STAGING=%SCRIPT_DIR%python3.installing"
Set "PYTHON_ARCHIVE=%TEMP%\python-%PYTHON_VERSION%-nuget.zip"
Set "PYTHON_URL=https://www.nuget.org/api/v2/package/python/%PYTHON_VERSION%"

if exist "%PYTHON_ROOT%\python.exe" (
    "%PYTHON_ROOT%\python.exe" -c "import importlib.util, sys; raise SystemExit(0 if sys.version_info[:3] == (3, 12, 10) and sys.maxsize.bit_length() == 63 and importlib.util.find_spec('ensurepip') is not None else 1)" >nul 2>nul
    if not errorlevel 1 (
        echo - Complete local Python 3.12.10 x64 already installed
        goto python_ready
    )
    echo - Replacing incomplete local Python runtime
)

echo - Installing Python %PYTHON_VERSION% x64 to %PYTHON_ROOT%
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ARCHIVE%'"
if errorlevel 1 (
    echo Failed to download Python %PYTHON_VERSION% x64
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $staging=[IO.Path]::GetFullPath('%PYTHON_STAGING%'); if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }; Expand-Archive -LiteralPath '%PYTHON_ARCHIVE%' -DestinationPath $staging -Force; $runtime=Join-Path $staging 'tools'; & (Join-Path $runtime 'python.exe') -c 'import ensurepip, ssl, sqlite3, sys; raise SystemExit(0 if sys.version_info[:3] == (3, 12, 10) and sys.maxsize.bit_length() == 63 else 1)'; if ($LASTEXITCODE -ne 0) { throw 'Complete Python runtime validation failed' }; $target=[IO.Path]::GetFullPath('%PYTHON_ROOT%'); if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }; Move-Item -LiteralPath $runtime -Destination $target; Remove-Item -LiteralPath $staging -Recurse -Force"
if errorlevel 1 (
    echo Failed to install Python %PYTHON_VERSION% x64
    pause
    exit /b 1
)

if exist "%PYTHON_ARCHIVE%" del /f /q "%PYTHON_ARCHIVE%" >nul 2>nul

if not exist "%PYTHON_ROOT%\python.exe" (
    echo Python installation completed but python.exe was not found at %PYTHON_ROOT%\python.exe
    pause
    exit /b 1
)

:python_ready

@REM Set paths
Set "PYTHON_EXE=%PYTHON_ROOT%\python.exe"

@REM Bootstrap pip from the complete runtime without another network download.
"%PYTHON_EXE%" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo - Installing pip into the local Python runtime
    "%PYTHON_EXE%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo Failed to install pip
        pause
        exit /b 1
    )
)

@REM Update pip
echo - Updating the pip package
"%PYTHON_EXE%" -m pip install --upgrade pip --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/ --no-warn-script-location
if %errorlevel% neq 0 (
    echo Failed to update pip
    pause
    exit /b 1
)

@REM Install requirements
echo.
echo - Installing the requirements
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements-cuda.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/ --no-warn-script-location
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/ --no-warn-script-location
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause
