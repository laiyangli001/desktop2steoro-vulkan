# Windows native launcher

`Desktop2Stereo.exe` is a native Win32 launcher. It displays `d2s_blur.png`
as a non-activating topmost startup window
before loading Python, starts `src/python3/python.exe`, and closes the layered
Splash after `src/desktop2stereo/logs/gui_ready.flag` is written.

Build from a Visual Studio Developer PowerShell:

```powershell
cmake -S native/launcher/windows -B native/launcher/windows/build -A x64
cmake --build native/launcher/windows/build --config Release
Copy-Item native/launcher/windows/build/Release/Desktop2Stereo.exe src/
```

The executable is placed in `src/` beside `python3/` and `desktop2stereo/`. It
also accepts a repository-root placement for compatibility and resolves all
runtime paths relative to its own location. License
checking is intentionally not implemented yet; the process boundary is ready
for the licensing state machine described in document 13.
