# Desktop2Stereo Linux launcher

`Desktop2Stereo` is a native, Python-independent X11-first launcher. Place the
executable beside the project root so that `src/python3/bin/python`,
`src/desktop2stereo/main.py`, and `src/desktop2stereo/d2s_blur.png` are found
relative to it. Build dependencies are X11, libpng, and pkg-config. Wayland
compositors may run the XWayland path; native transparent Wayland behavior is
still marked as pending validation.
