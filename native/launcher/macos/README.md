# Desktop2Stereo macOS launcher

`Desktop2Stereo-macos` is a standalone AppKit executable. It shows a borderless
transparent splash before launching the project Python runtime. The release
layout places this executable directly in `src/`, beside `python3/` and
`desktop2stereo/`; CMake can produce a Universal (`x86_64;arm64`) build.

The native process owns only the splash and Python process boundary. Python
opens the independent Flet authorization GUI before loading GUI1 or GUI2;
`auth_ready.flag` releases the splash when the login window is visible and
`gui_ready.flag` represents runtime GUI readiness. Native licensing logic is
intentionally kept out of the splash binary.
