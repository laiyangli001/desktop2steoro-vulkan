#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(D2S_NVFRUC_BUILD)
#    define D2S_NVFRUC_API __declspec(dllexport)
#  else
#    define D2S_NVFRUC_API __declspec(dllimport)
#  endif
#else
#  define D2S_NVFRUC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* d2s_nvfruc_handle;

D2S_NVFRUC_API uint32_t d2s_nvfruc_abi_version(void);
D2S_NVFRUC_API int32_t d2s_nvfruc_probe(void);
D2S_NVFRUC_API const char* d2s_nvfruc_last_error(void);

D2S_NVFRUC_API d2s_nvfruc_handle d2s_nvfruc_create(
    uint32_t width,
    uint32_t height,
    int32_t device_index,
    uint64_t cuda_stream
);

D2S_NVFRUC_API int32_t d2s_nvfruc_process(
    d2s_nvfruc_handle handle,
    uint64_t previous_ptr,
    size_t previous_pitch,
    double previous_timestamp,
    uint64_t next_ptr,
    size_t next_pitch,
    double next_timestamp,
    uint64_t output_ptr,
    size_t output_pitch,
    double output_timestamp,
    uint64_t cuda_stream
);

D2S_NVFRUC_API int32_t d2s_nvfruc_reset(d2s_nvfruc_handle handle);
D2S_NVFRUC_API void d2s_nvfruc_destroy(d2s_nvfruc_handle handle);

#ifdef __cplusplus
}
#endif
