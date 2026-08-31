# Desktop2Stereo macOS launcher

`Desktop2Stereo.app` uses AppKit to show a borderless transparent Splash before
launching the bundled project Python runtime. The release layout places the
app bundle beside `src/`; the launcher resolves `src/python3/bin/python`,
`src/desktop2stereo/main.py`, and `d2s_blur.png` without relying on the caller's
working directory. CMake can produce a Universal (`x86_64;arm64`) build.

Native licensing checks remain an extension point and are not implemented in
this bootstrap binary yet.
