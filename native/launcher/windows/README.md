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
runtime paths relative to its own location. The native process only owns the
startup artwork and process boundary; Python starts the independent Flet
authorization GUI first, then loads GUI1 or GUI2 only after authorization.
`auth_ready.flag` releases the artwork after the login window is visible and
`gui_ready.flag` is reserved for the selected runtime GUI.
