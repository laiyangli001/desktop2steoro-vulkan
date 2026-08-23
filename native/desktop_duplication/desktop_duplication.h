#pragma once
#ifdef _WIN32
#include <stdint.h>
#ifdef D2S_DESKTOP_DUPLICATION_BUILD
#define D2S_DESKTOP_DUPLICATION_API __declspec(dllexport)
#else
#define D2S_DESKTOP_DUPLICATION_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_probe(void);
D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_last_error(char*, int);
D2S_DESKTOP_DUPLICATION_API void* d2s_desktop_duplication_create(int, int);
D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_acquire(void*, void**, int*, int*, uint64_t*);
D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_copy_frame(void*, unsigned char*, int, int*, int*, int*);
D2S_DESKTOP_DUPLICATION_API void* d2s_desktop_duplication_device(void*);
D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_release_frame(void*);
D2S_DESKTOP_DUPLICATION_API void d2s_desktop_duplication_destroy(void*);
#ifdef __cplusplus
}
#endif
#endif
