#pragma once

#ifdef _WIN32
#ifdef D2S_OPENVINO_D3D11_BUILD
#define D2S_OPENVINO_D3D11_API __declspec(dllexport)
#else
#define D2S_OPENVINO_D3D11_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Capability bits: bit 0 = native NV12 surface RemoteTensor; bit 1 = GPU BGRA->NV12 conversion.
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_capabilities(void);
D2S_OPENVINO_D3D11_API void* d2s_openvino_d3d11_create(const char* model_path, void* d3d11_device);
// Returns the DXGI adapter LUID packed as HighPart:LowPart, or 0 on failure.
D2S_OPENVINO_D3D11_API unsigned long long d2s_openvino_d3d11_adapter_luid(void* handle);
// Returns a borrowed NV12 D3D11 texture produced by the latest set_texture call.
// The pointer remains owned by the bridge and is invalidated by the next frame or destroy.
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_nv12_surface(
    void* handle, void** texture, int* width, int* height);
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_set_texture(void* handle, const char* input_name, void* d3d11_texture, int width, int height);
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_infer(void* handle);
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_output_shape(void* handle, long long* dims, int capacity);
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_read_output(void* handle, float* output, int capacity);
D2S_OPENVINO_D3D11_API int d2s_openvino_d3d11_last_error(char* output, int capacity);
D2S_OPENVINO_D3D11_API void d2s_openvino_d3d11_destroy(void* handle);
#ifdef __cplusplus
}
#endif
#endif
