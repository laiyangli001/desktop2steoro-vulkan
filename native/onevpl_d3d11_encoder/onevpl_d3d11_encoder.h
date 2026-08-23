#pragma once

#ifdef _WIN32
#define D2S_ONEVPL_API __declspec(dllexport)
#else
#define D2S_ONEVPL_API
#endif

extern "C" {

// Returns 1 only when the bridge was built with oneVPL headers/library and a
// usable Intel D3D11 implementation is available.
D2S_ONEVPL_API int d2s_onevpl_d3d11_probe();
D2S_ONEVPL_API int d2s_onevpl_d3d11_last_error(char* output, int capacity);

// The device and texture are borrowed from the caller. The texture must be
// an NV12 ID3D11Texture2D created by the same adapter as the device.
D2S_ONEVPL_API void* d2s_onevpl_d3d11_create(
    int width, int height, int fps, int bitrate, int hevc, void* d3d11_device);
D2S_ONEVPL_API unsigned long long d2s_onevpl_d3d11_adapter_luid(void* handle);
D2S_ONEVPL_API int d2s_onevpl_d3d11_submit_nv12(
    void* handle, void* nv12_texture, long long timestamp);
D2S_ONEVPL_API int d2s_onevpl_d3d11_read_packet(
    void* handle, void* output, int capacity);
D2S_ONEVPL_API void d2s_onevpl_d3d11_destroy(void* handle);

}
