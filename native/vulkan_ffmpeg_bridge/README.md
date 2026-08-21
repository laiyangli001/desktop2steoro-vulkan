# Vulkan FFmpeg bridge

This directory defines the in-process bridge required by the final Advanced
Network Streaming Vulkan path. The bridge now creates an FFmpeg-owned Vulkan
NV12 frame pool and exposes the pool's GPU image handles through ABI version 2.
The caller writes those images with Vulkan/CUDA interop and submits the same
GPU frame to FFmpeg through `AV_PIX_FMT_VULKAN`; it must not send RGB24 bytes
or a CPU pointer.

The local application deliberately falls back to the validated host-upload
path until the Python frame-pool consumer, GPU RGB-to-NV12 conversion,
semaphore wait, and Windows GPU artifact have passed a real 4K headset test.

Build remotely with:

```text
GitHub Actions → Build Vulkan FFmpeg Bridge → Windows x86_64
```

The workflow downloads the pinned FFmpeg package from
`laiyangli001/desktop2stereo-ffmpeg-builds`, builds with MinGW and Vulkan
headers, and verifies the exported ABI. The resulting DLL is an artifact only;
it must not be copied into the application until image import, semaphore
synchronization, packet output, and 4K WebRTC acceptance are complete.
