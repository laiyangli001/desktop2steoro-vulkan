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

}

