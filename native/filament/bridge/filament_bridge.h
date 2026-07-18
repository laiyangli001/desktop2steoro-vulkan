#pragma once

#include <stdint.h>

#if defined(_WIN32)
#if defined(FILAMENT_BRIDGE_BUILD)
#define FILAMENT_BRIDGE_API __declspec(dllexport)
#else
#define FILAMENT_BRIDGE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define FILAMENT_BRIDGE_API __attribute__((visibility("default")))
#else
#define FILAMENT_BRIDGE_API
#endif

extern "C" {

struct FilamentBridge;

// The caller must have made the OpenXR OpenGL context current on this thread.
FILAMENT_BRIDGE_API FilamentBridge* filament_bridge_create(void* shared_gl_context);
FILAMENT_BRIDGE_API void filament_bridge_destroy(FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_load_glb(FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_bridge_apply_animations(FilamentBridge* bridge, double time_seconds);
FILAMENT_BRIDGE_API uint32_t filament_bridge_animation_count(const FilamentBridge* bridge);
FILAMENT_BRIDGE_API float filament_bridge_animation_duration(const FilamentBridge* bridge, uint32_t animation_index);
FILAMENT_BRIDGE_API const char* filament_bridge_last_error(const FilamentBridge* bridge);

}