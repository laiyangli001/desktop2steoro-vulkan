@echo off
echo --- Desktop2Stereo Installer (With CUDA for NVIDIA GPUs.) ---
echo - Setting up the virtual environment

@REM Install local Python 3.12 x64 first
Set "PYTHON_VERSION=3.12.10"
Set "SCRIPT_DIR=%~dp0"
Set "PYTHON_ROOT=%SCRIPT_DIR%python3"
Set "PYTHON_INSTALLER=%TEMP%\python-%PYTHON_VERSION%-amd64.exe"
Set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe"

if exist "%PYTHON_ROOT%\python.exe" (
    "%PYTHON_ROOT%\python.exe" -c "import platform, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and platform.architecture()[0] == '64bit' else 1)"
    if not errorlevel 1 (
        echo - Local Python 3.12 x64 already installed
        goto python_ready
    )
)

echo - Installing Python %PYTHON_VERSION% x64 to %PYTHON_ROOT%
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"
if errorlevel 1 (
    echo Failed to download Python %PYTHON_VERSION% x64
    pause
    exit /b 1
)

start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 TargetDir="%PYTHON_ROOT%" Include_pip=1 Include_launcher=0 PrependPath=0 Include_test=0 Shortcuts=0 AssociateFiles=0
if errorlevel 1 (
    echo Failed to install Python %PYTHON_VERSION% x64
    pause
    exit /b 1
)

if exist "%PYTHON_INSTALLER%" del /f /q "%PYTHON_INSTALLER%" >nul 2>nul

if not exist "%PYTHON_ROOT%\python.exe" (
    echo Python installation completed but python.exe was not found at %PYTHON_ROOT%\python.exe
    pause
    exit /b 1
)

:python_ready

@REM Set paths
Set "PYTHON_EXE=%PYTHON_ROOT%\python.exe"


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
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/ --no-warn-script-location
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements-cuda0.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host https://repo.huaweicloud.com/ --no-warn-script-location
if %errorlevel% neq 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment deployed successfully.
pause