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
6. Keep the licensing check as an injectable interface. No license state,
   secret, or token is implemented in this bootstrap layer yet.

The ready flag is a startup handshake only; it is not an authorization proof.

`launcher_contract.h` defines the stable status values reserved for future
licensing integration. Platform launchers may add an injected
`D2SLicenseCheck` implementation later; the current binaries do not perform a
license check and do not contain credentials.
