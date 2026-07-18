@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%python3\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%python3\python.exe"
) else (
    set "PYTHON_EXE=python"
)
"%PYTHON_EXE%" "%SCRIPT_DIR%main.py" %*
exit /b %ERRORLEVEL%

