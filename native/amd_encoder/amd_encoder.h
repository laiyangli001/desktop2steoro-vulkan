#pragma once

#ifdef _WIN32
#define D2S_AMD_API __declspec(dllexport)
#else
#define D2S_AMD_API
#endif

extern "C" {

// Returns 1 when the AMD AMF runtime and a D3D11 device can be created.
D2S_AMD_API int d2s_amd_encoder_probe();

// Writes a UTF-8 diagnostic into caller-provided storage.
D2S_AMD_API int d2s_amd_encoder_last_error(char* output, int capacity);

// Create an AMF encoder that accepts D3D11 NV12 textures. Returns an opaque
// handle, or nullptr when the AMF SDK/runtime is unavailable.
D2S_AMD_API void* d2s_amd_encoder_create(int width, int height, int fps, int bitrate, int hevc);
D2S_AMD_API int d2s_amd_encoder_submit_texture(void* handle, void* d3d11_texture);
D2S_AMD_API int d2s_amd_encoder_read_packet(void* handle, void* output, int capacity);
D2S_AMD_API void d2s_amd_encoder_destroy(void* handle);

}
