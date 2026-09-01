# Cross-platform launcher contract

All platform launchers follow the same contract:

1. Resolve resources relative to the launcher executable or application bundle.
2. Show the platform-native Splash before starting Python.
3. Start the project entry point without requiring the caller's working
   directory or a shell script.
4. Wait for `src/desktop2stereo/logs/gui_ready.flag` (or the equivalent
   release-layout path) before closing Splash.
5. Monitor the child process and report startup failures with a platform-native
   error dialog.
6. Keep licensing in the Python/Flet authorization gate. The native bootstrap
   layer only starts that gate, observes its startup handshake, and never
   stores license state, secrets, or tokens.

The ready flag is a startup handshake only; it is not an authorization proof.

`launcher_contract.h` defines stable startup status values. The current
platform launchers deliberately do not implement licensing themselves: they
start `desktop2stereo.main`, which runs the independent Flet authorization GUI
and the Python-side double-check before importing GUI1 or GUI2.
