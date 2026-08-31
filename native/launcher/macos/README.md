# macOS launcher placeholder

The macOS launcher is planned as an AppKit `.app` with a borderless,
transparent `NSWindow`. Separate Intel and Apple Silicon builds, plus a
Universal package, must be validated before release. It must implement the
common launcher contract in `../common/README.md` and must not depend on
Python for the Splash window.
