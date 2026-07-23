@echo off
setlocal

set "ROOM=%~1"
if "%ROOM%"=="" set "ROOM=3D_Home"

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%.."
set "PY=%APP_DIR%\python3\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python not found: "%PY%"
  pause
  exit /b 1
)

pushd "%APP_DIR%"
set "PYTHONPATH=%APP_DIR%"
"%PY%" "%SCRIPT_DIR%preview_room_layout.py" "%ROOM%"
set "RESULT=%ERRORLEVEL%"
popd
if not "%RESULT%"=="0" (
  echo [ERROR] Room layout preview failed.
  pause
  exit /b 1
)
