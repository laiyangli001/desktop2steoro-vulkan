@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
pushd "%~dp0"
set ROOT=%CD%\..\..\..
"C:\Program Files\CMake\bin\cmake.exe" -S . -B "%ROOT%\native\filament\build\windows-nmake" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\CMake\bin\cmake.exe" --build "%ROOT%\native\filament\build\windows-nmake" --parallel 4
popd