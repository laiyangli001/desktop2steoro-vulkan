# Desktop2Stereo macOS launcher

`Desktop2Stereo.app` uses AppKit to show a borderless transparent Splash before
launching the bundled project Python runtime. The release layout places the
app bundle in `src/` beside `python3/` and `desktop2stereo/`; a bundle at the
repository root is accepted for compatibility. CMake can produce a Universal
(`x86_64;arm64`) build.

Native licensing checks remain an extension point and are not implemented in
this bootstrap binary yet.
