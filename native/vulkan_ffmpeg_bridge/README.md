# Vulkan FFmpeg bridge

This directory defines the in-process bridge used by the Advanced Network
Streaming Vulkan path. The bridge creates an FFmpeg-owned Vulkan device (with
Vulkan Video extensions), an RGBA input pool and a single NV12 multi-plane
encode pool, then exposes GPU image handles, external memory handles and
timeline semaphore values through ABI version 5.
Before exporting a frame, the bridge transitions it to `GENERAL` on FFmpeg's
Vulkan queue and advances its timeline semaphore. The caller waits that value,
writes those images with Vulkan/CUDA interop, signals the next value, and submits the same
GPU frame to FFmpeg through `AV_PIX_FMT_VULKAN`; it must not send RGB24 bytes
or a CPU pointer. Passing an application-owned `VkDevice` is optional and is
allowed only when that device was created with the required Vulkan Video
extensions.

The application now enables this bridge when the ABI, FFmpeg DLLs, Vulkan
probe and CUDA importer are available. The normal path performs CUDA RGBA →
Vulkan Compute RGBA-to-NV12 → `h264_vulkan`/`hevc_vulkan`; compressed packets
only are sent to the muxer. Any initialization or runtime failure logs the
layer and falls back to the stable host-upload advanced streaming path.

## Current CUDA limitation

The CUDA-friendly `AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE` pool exports two
single-plane images. Validation on NVIDIA RTX 3090 shows these images are not
legal `h264_vulkan` / `hevc_vulkan` input resources: Vulkan Video requires one
NV12 multi-plane image. The bridge is therefore a synchronization and external
handle diagnostic only. The split representation is rejected by the Python
frame gate and never submitted as a Vulkan Video source. The enabled application path writes an FFmpeg-owned single-plane RGBA
image with CUDA, performs the color conversion on the same Vulkan device and
submits `AV_PIX_FMT_VULKAN`. At startup it queries whether the exact NV12
encode image supports `STORAGE_IMAGE`; supported drivers write the two plane
views directly and log `zero_copy=True`. Drivers without that capability use
the bounded device-local RGBA/R8/RG8-to-NV12 copy and log
`zero_copy=False`. Both variants remove the CPU download and raw RGB24 pipe.

Build remotely with:

```text
GitHub Actions → Build Vulkan FFmpeg Bridge → Windows x86_64
```

The workflow downloads the pinned FFmpeg package from
`laiyangli001/desktop2stereo-ffmpeg-builds`, builds with MinGW and Vulkan
headers, and verifies the exported ABI. The resulting DLL is an artifact only;
it must not be copied into the application until image import, semaphore
synchronization, packet output, and 4K WebRTC acceptance are complete.
