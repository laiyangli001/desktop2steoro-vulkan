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

// Handles are borrowed from the Python-owned OpenXR Vulkan session.
struct FilamentBridgeVulkanCreateInfo {
    void* instance;
    void* physical_device;
    void* device;
    uint32_t graphics_queue_family_index;
    uint32_t graphics_queue_index;
};

FILAMENT_BRIDGE_API FilamentBridge* filament_bridge_create_vulkan(
        const FilamentBridgeVulkanCreateInfo* info);
FILAMENT_BRIDGE_API void filament_bridge_destroy(FilamentBridge* bridge);

// image_handles points to borrowed VkImage handles owned by an OpenXR swapchain.
FILAMENT_BRIDGE_API int filament_bridge_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_set_acquired_image(
        FilamentBridge* bridge, uint32_t image_index);
FILAMENT_BRIDGE_API int filament_bridge_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z);
FILAMENT_BRIDGE_API int filament_bridge_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_bridge_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_bridge_begin_frame(FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_end_frame(FilamentBridge* bridge);

FILAMENT_BRIDGE_API int filament_bridge_load_glb(
        FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_bridge_apply_animations(
        FilamentBridge* bridge, double time_seconds);
FILAMENT_BRIDGE_API uint32_t filament_bridge_animation_count(
        const FilamentBridge* bridge);
FILAMENT_BRIDGE_API float filament_bridge_animation_duration(
        const FilamentBridge* bridge, uint32_t animation_index);
FILAMENT_BRIDGE_API const char* filament_bridge_last_error(
        const FilamentBridge* bridge);

}
