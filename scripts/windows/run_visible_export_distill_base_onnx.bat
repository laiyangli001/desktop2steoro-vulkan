@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0..\.."
set "PYTHON_EXE=%LAB_DIR%\python3\python.exe"
set "ONNX_DIR=%LAB_DIR%\models\models--lc700x--Distill-Any-Depth-Base-hf"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  echo [Hint] Copy Desktop2Stereo\python3 into this lab as python3 first.
  pause
  exit /b 1
)

title 4K Stereo Lab - Export Distill ONNX
echo [Info] Export target folder: %ONNX_DIR%
echo [Info] Model: Distill-Any-Depth-Base
echo [Info] Model ID: lc700x/Distill-Any-Depth-Base-hf
echo [Info] ONNX input: 1x3x294x518
echo [Info] DType: auto
echo [Info] The script forces network-enabled model download/cache by default.
echo [Info] This may take several minutes.
echo.

pushd "%LAB_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\tools\export_distill_base_onnx.py" --device cuda --dtype auto
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening ONNX output folder ...
  explorer "%ONNX_DIR%"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
