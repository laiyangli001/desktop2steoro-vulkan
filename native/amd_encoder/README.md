# Windows AMD AMF bridge

This optional DLL is the first stage of the Windows AMD GPU-surface encoder.
It probes the installed AMD AMF runtime (`amfrt64.dll`) and the AMD DXGI
adapter without changing the existing FFmpeg audio/MediaMTX path.

Build with Visual Studio or the Windows SDK:

```powershell
cmake -S native/amd_encoder -B build/amd_encoder -A x64
cmake --build build/amd_encoder --config Release
```

Copy `d2s_amd_encoder.dll` to `src/desktop2steoro/streaming/amd_encoder/` or set
`D2S_AMD_ENCODER_DLL` to its absolute path.

The actual HIP-to-D3D11 shared-surface submission and AMF packet encoder is
the next layer; until it is present, the application must not label the
FFmpeg rawvideo path as AMD zero-copy.
