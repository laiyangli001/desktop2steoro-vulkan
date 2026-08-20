@echo off
setlocal
cd /d "%~dp0\..\.."
echo Building native TensorRT engine. First build may take several minutes.
echo.
for /f "tokens=1,* delims==" %%A in (project_paths.env) do if "%%A"=="D2S_PYTHON_DIR" set "D2S_PYTHON_DIR=%%B"
set "PYTHON_EXE=%D2S_PYTHON_DIR%\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)
"%PYTHON_EXE%" -B scripts\tools\build_native_tensorrt_engine.py %*
echo.
pause
