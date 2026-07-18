@echo off
setlocal
pushd "%~dp0"
set ROOT=%~dp0..\..\..
set SDK=%ROOT%\native\filament\sdk\windows\v1.74.0
set OUT=%ROOT%\src\xr_viewer\native
if not exist "%OUT%" mkdir "%OUT%"
set LIB=%SDK%\lib\x86_64\md
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
setlocal EnableDelayedExpansion
set LIBS=
for /f "delims=" %%F in ('dir /b /s "%LIB%\*.lib"') do set LIBS=!LIBS! "%%F"
cl /nologo /std:c++20 /EHsc /MD /LD /I "%SDK%\include" filament_bridge.cpp /link !LIBS! opengl32.lib user32.lib gdi32.lib shlwapi.lib /OUT:"%OUT%\filament_bridge.dll"
endlocal
popd
