#pragma once

#ifdef _WIN32
#define D2S_D3D11_SURFACE_API __declspec(dllexport)
#else
#define D2S_D3D11_SURFACE_API
#endif

// Optional final-SBS surface bridge. The caller uploads the already composed
// SBS RGB frame; the bridge owns the D3D11 BGRA/NV12 surfaces and exposes the
// borrowed NV12 texture to a same-device oneVPL encoder.
extern "C" {
D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_probe(void);
D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_create(int width, int height, int adapter_index);
D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_create_from_device(
    int width, int height, void* d3d11_device);
D2S_D3D11_SURFACE_API void* d2s_d3d11_sbs_surface_device(void* handle);
D2S_D3D11_SURFACE_API unsigned long long d2s_d3d11_sbs_surface_adapter_luid(void* handle);
D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_set_bgra_texture(
    void* handle, void* bgra_texture, unsigned long long adapter_luid);
D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_upload_bgra(
    void* handle, const unsigned char* data, int stride, int width, int height);
D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_nv12(
    void* handle, void** texture, int* width, int* height);
D2S_D3D11_SURFACE_API int d2s_d3d11_sbs_surface_last_error(char* output, int capacity);
D2S_D3D11_SURFACE_API void d2s_d3d11_sbs_surface_destroy(void* handle);
}
