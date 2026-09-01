# Desktop2Stereo Linux launcher

`Desktop2Stereo` is a native, Python-independent X11-first launcher. Place the
executable in `src/` beside `python3/` and `desktop2stereo/`; it also accepts a
repository-root placement for compatibility. Build dependencies are X11, libpng, and pkg-config. Wayland
compositors may run the XWayland path; native transparent Wayland behavior is
still marked as pending validation. The launcher starts the Python
authorization gate and accepts `auth_ready.flag` for the login window or
`gui_ready.flag` for the selected runtime GUI; authorization failure never
reports runtime readiness.
