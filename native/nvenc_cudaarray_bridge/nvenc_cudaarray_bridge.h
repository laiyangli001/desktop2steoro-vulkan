#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(D2S_NVENC_CUDAARRAY_BUILD)
#    define D2S_NVENC_API __declspec(dllexport)
#  else
#    define D2S_NVENC_API __declspec(dllimport)
#  endif
#else
#  define D2S_NVENC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* d2s_nvenc_cudaarray_handle;

D2S_NVENC_API uint32_t d2s_nvenc_cudaarray_abi_version(void);
D2S_NVENC_API int32_t d2s_nvenc_cudaarray_probe(void);

D2S_NVENC_API d2s_nvenc_cudaarray_handle d2s_nvenc_cudaarray_create(
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    uint32_t bitrate,
    int32_t hevc,
    uint64_t cuda_array
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_submit(
    d2s_nvenc_cudaarray_handle handle,
    int64_t timestamp
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_read_packet(
    d2s_nvenc_cudaarray_handle handle,
    uint8_t* destination,
    size_t capacity,
    size_t* packet_size
);

D2S_NVENC_API int32_t d2s_nvenc_cudaarray_flush(
    d2s_nvenc_cudaarray_handle handle
);

D2S_NVENC_API void d2s_nvenc_cudaarray_destroy(
    d2s_nvenc_cudaarray_handle handle
);

D2S_NVENC_API const char* d2s_nvenc_cudaarray_last_error(void);

#ifdef __cplusplus
}
#endif
