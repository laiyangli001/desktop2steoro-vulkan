@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0..\.."
set "PYTHON_EXE=%LAB_DIR%\python3\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  echo [Hint] Copy Desktop2Stereo\python3 into this lab as python3 first.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo [Usage] Drag one RGB image onto this .bat, or run:
  echo         %~nx0 rgb.png [reference_depth.png]
  pause
  exit /b 1
)

set "RGB=%~1"
set "REF_DEPTH=%~2"
set "OUT_DIR=%LAB_DIR%\outputs\depth_compare"

title 4K Stereo Lab - Distill Depth Map
echo [Info] RGB: %RGB%
echo [Info] Output: %OUT_DIR%
echo [Info] Depth model: Distill-Any-Depth-Base @ 518
echo [Info] Backend priority: TensorRT -^> ONNX CUDA IOBinding -^> PyTorch CUDA
echo [Info] Model ID: lc700x/Distill-Any-Depth-Base-hf
echo [Info] This is the most direct 3D comparison: generated depth map first.
if not "%REF_DEPTH%"=="" echo [Info] Reference depth: %REF_DEPTH%
echo [Info] First torch/CUDA import or model download may take several minutes.
echo.

pushd "%LAB_DIR%"
if "%REF_DEPTH%"=="" (
  "%PYTHON_EXE%" "%LAB_DIR%\scripts\tools\generate_depth_map.py" --rgb "%RGB%" --provider distill_base_nvidia --out-dir "%OUT_DIR%" --device cuda
) else (
  "%PYTHON_EXE%" "%LAB_DIR%\scripts\tools\generate_depth_map.py" --rgb "%RGB%" --provider distill_base_nvidia --reference-depth "%REF_DEPTH%" --out-dir "%OUT_DIR%" --device cuda
)
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
  echo.
  echo [Info] Opening outputs\depth_compare ...
  explorer "%OUT_DIR%"
)
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
