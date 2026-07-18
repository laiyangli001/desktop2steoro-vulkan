@echo off
setlocal enabledelayedexpansion

:: Get the directory where this batch file is located
set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
cd /d "%APP_DIR%"

:: Set variables
set REPO_OWNER=lc700x
set REPO_NAME=desktop2stereo
set BRANCH=update
set ZIP_FILE=%REPO_NAME%-%BRANCH%.zip

:: Construct the download URL
set DOWNLOAD_URL=https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%BRANCH%.zip

:: Download the ZIP archive
echo Downloading update from %DOWNLOAD_URL%
curl -L -o "%ZIP_FILE%" "%DOWNLOAD_URL%"
if errorlevel 1 (
    echo Failed to download update.
    pause
    exit /b 1
)

:: Extract using PowerShell's Expand-Archive (safe and reliable)
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath './temp_extract' -Force"
if errorlevel 1 (
    echo Failed to extract update.
    pause
    exit /b 1
)

:: Move contents from nested folder to current directory
for /d %%F in (temp_extract\*) do xcopy "%%F\*" ".\" /E /H /Y

:: Clean up
rmdir /S /Q temp_extract
:: Remove unnecessary platform folders
rmdir /S /Q src\streaming\rtmp\mac 2>nul
rmdir /S /Q src\streaming\rtmp\linux 2>nul
del /F /Q update_mac_linux 2>nul
del /F /Q update.bat 2>nul
del /F /Q main 2>nul
del "%ZIP_FILE%"

echo Latest Desktop2Stereo downloaded and extracted to current folder.
@REM Set paths
set "PYTHON_EXE=%APP_DIR%\src\python3\python.exe"
if not exist "!PYTHON_EXE!" (
    echo [Error] Python not found at !PYTHON_EXE!
    echo The Desktop2Stereo workspace seems incomplete.
    pause
    exit /b 1
)

@REM Update pip
echo [Desktop2Stereo Patch]

@REM Install new requirements
echo.
echo - Installing new requirements
"!PYTHON_EXE!" -m pip uninstall pyaudio -y 2>nul
"!PYTHON_EXE!" -m pip install -r "%APP_DIR%\requirements.txt" --no-cache-dir --no-warn-script-location -i https://repo.huaweicloud.com/repository/pypi/simple/ --trusted-host repo.huaweicloud.com
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Python environment updated successfully.
pause
exit /b 0
