# Vulkan FFmpeg bridge

This directory defines the in-process bridge required by the final Advanced
Network Streaming Vulkan path. The bridge receives an already synchronized
Vulkan encode-source image and submits it to FFmpeg through `AV_PIX_FMT_VULKAN`.
It must not receive RGB24 frame bytes or a CPU pointer.

The local application deliberately falls back to the validated host-upload
path until the `submit_image` implementation is complete and the Windows GPU
artifact has passed a real 4K headset test.

Build remotely with:

```text
GitHub Actions → Build Vulkan FFmpeg Bridge → Windows x86_64
```

The workflow downloads the pinned FFmpeg package from
`laiyangli001/desktop2stereo-ffmpeg-builds`, builds with MinGW and Vulkan
headers, and verifies the exported ABI. The resulting DLL is an artifact only;
it must not be copied into the application until image import, semaphore
synchronization, packet output, and 4K WebRTC acceptance are complete.

