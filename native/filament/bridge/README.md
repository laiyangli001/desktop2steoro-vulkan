# Filament Vulkan bridge

This directory contains the only project-owned native boundary. It creates a
Filament Vulkan Engine from the Vulkan handles supplied by the Python OpenXR
session, exposes OpenXR VkImages as a Filament SwapChain, and loads GLB assets
through Filament gltfio. It never accepts or returns CPU pixel buffers.

The OpenXR application owns the Vulkan instance, device, queue, swapchain image
acquire/release, and frame pacing. The bridge only borrows those handles and
lets Filament render into the acquired image. The bridge never calls
`vkAcquireNextImageKHR`, `vkQueuePresentKHR`, or destroys an OpenXR object.
One bridge owns one Filament Engine and shared Scene; its two eye targets own
independent Filament Renderers, Views, Cameras, and external swapchains. The
separate Renderers keep Filament frame state isolated while GLB, materials,
textures, and shader resources remain shared by the Engine.

Each target platform needs its matching official Filament SDK archive. Configure
CMake with `FILAMENT_SDK_ROOT` pointing at that extracted archive. The generated
library is placed in `src/xr_viewer/native` for packaging:

```text
Windows: filament_bridge.dll
macOS:   libfilament_bridge.dylib
Linux:   libfilament_bridge.so
```

The C ABI in `filament_bridge.h` is intentionally narrow. Python first creates
one bridge with `filament_bridge_create_vulkan`, registers both eye swapchains
with `filament_bridge_create_eye_swapchain`, selects the eye with
`filament_bridge_set_active_eye`, calls `filament_bridge_set_acquired_image`,
then brackets Filament rendering with `filament_bridge_begin_frame` and
`filament_bridge_end_frame`. GLB and controller assets are loaded once into the
shared Scene; each eye's active Renderer brackets only its own external
Swapchain frame.

The desktop preview ABI exposes `filament_preview_apply_animations` for embedded glTF animations.

The OpenXR bridge prints bounded native diagnostics for the first eight stereo
frames, including eye, acquired image index, raw image handle, Renderer, and
Swapchain pointers. These messages are intended to identify external image
lifetime failures without enabling a per-frame verbose log mode.

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
