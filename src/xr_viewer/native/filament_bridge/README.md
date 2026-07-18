# Filament bridge

This directory contains the portable C++ bridge. It loads GLB assets through
Filament gltfio and applies standard glTF animation channels through Animator.
It never accepts or returns CPU pixel buffers.

Each target platform needs its matching official Filament SDK archive. Configure
CMake with `FILAMENT_SDK_ROOT` pointing at that extracted archive. The generated
library is placed in the parent `src/xr_viewer/native` directory for packaging:

```text
Windows: filament_bridge.dll
macOS:   libfilament_bridge.dylib
Linux:   libfilament_bridge.so
```

OpenXR context and swapchain binding is deliberately platform-specific and is
not implemented by this portable asset and animation layer.

GitHub Actions builds the Windows x86_64, Linux x86_64, and macOS arm64 release libraries from
the matching official SDK archives, validates their SHA-256 digests, and uploads
the resulting library as a workflow artifact.

## SDK upgrades

`../version.json` is the single source of truth for the validated Filament
release and the SHA-256 digests of its Windows, Linux, and macOS SDK archives.
The `Watch Filament Releases` workflow checks upstream each Monday and creates
one issue for a newly published version. It never changes runtime binaries.

To upgrade, update that manifest from the official release, then run `Build
Filament Bridge`. Merge only after all three platform jobs pass; download their
artifacts into `src/xr_viewer/native` and commit those three libraries together
with the manifest change. Reverting that commit restores the last verified SDK
and binaries.
