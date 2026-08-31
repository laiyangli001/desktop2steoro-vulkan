# Windows native launcher

`Desktop2Stereo.exe` is a native Win32 launcher. It displays `d2s_blur.png`
before loading Python, starts `src/python3/python.exe`, and closes the layered
Splash after `src/desktop2stereo/logs/gui_ready.flag` is written.

Build from a Visual Studio Developer PowerShell:

```powershell
cmake -S native/launcher/windows -B native/launcher/windows/build -A x64
cmake --build native/launcher/windows/build --config Release
Copy-Item native/launcher/windows/build/Release/Desktop2Stereo.exe .
```

The executable can be placed in the repository root or the root of a release
directory. It resolves all runtime paths relative to its own location. License
checking is intentionally not implemented yet; the process boundary is ready
for the licensing state machine described in document 13.
