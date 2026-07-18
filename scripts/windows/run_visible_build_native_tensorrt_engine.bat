@echo off
setlocal
cd /d "%~dp0\.."
echo Building native TensorRT engine. First build may take several minutes.
echo.
python3\python.exe -B scripts\tools\build_native_tensorrt_engine.py %*
echo.
pause
