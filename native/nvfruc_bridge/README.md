# d2s_nvfruc_bridge

This directory contains the optional Windows native bridge used by the
process-local NvFRUC stage.

The NVIDIA Optical Flow SDK is not included in this repository. Obtain the SDK
under its applicable license and pass its root to CMake:

```powershell
cmake -S . -B build -DD2S_NVFRUC_SDK_ROOT=C:\path\to\NvOFFRUC-SDK
cmake --build build --config Release
```

The SDK root must contain `include/NvOFFRUC.h`. At runtime, place the licensed
`NvOFFRUC.dll` beside the bridge or on the normal Windows DLL search path.
The bridge dynamically resolves `NvOFFRUCCreate`,
`NvOFFRUCRegisterResource`, `NvOFFRUCProcess`, and
`NvOFFRUCDestroy`; it does not link against or redistribute the SDK runtime.

The C ABI accepts CUDA device pointers and uses CUDA arrays internally. No CPU
image buffer is created by this bridge. Build and package the SDK runtime only
according to NVIDIA's redistribution terms.
