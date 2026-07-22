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
struct FilamentPreview;

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
FILAMENT_BRIDGE_API int filament_bridge_create_eye_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_set_active_eye(
        FilamentBridge* bridge, uint32_t eye_index);
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
// Submit queued work without blocking; wait once after the complete XR frame.
FILAMENT_BRIDGE_API int filament_bridge_wait_for_idle(FilamentBridge* bridge);

FILAMENT_BRIDGE_API int filament_bridge_load_glb(
        FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count);

// Controller assets are loaded into the same Filament scene as the environment.
FILAMENT_BRIDGE_API int filament_bridge_load_controller(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_pose(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip,
        float joystick_x, float joystick_y,
        uint32_t button_mask);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_visible(
        FilamentBridge* bridge, uint32_t hand, int visible);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_laser(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16, int visible);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_guide_texture(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_guide(
        FilamentBridge* bridge, const float* matrix16, int visible);

FILAMENT_BRIDGE_API int filament_bridge_set_scene_exposure(
        FilamentBridge* bridge, float exposure_ev);
FILAMENT_BRIDGE_API int filament_bridge_set_skybox_brightness(
        FilamentBridge* bridge, float brightness);
FILAMENT_BRIDGE_API int filament_bridge_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z);
FILAMENT_BRIDGE_API int filament_bridge_create_screen(FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_set_screen(
        FilamentBridge* bridge,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees);
// image is a borrowed Vulkan VkImage owned by the runtime output adapter.
FILAMENT_BRIDGE_API int filament_bridge_set_screen_image(
        FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format);
// Set a borrowed binary semaphore signaled by the runtime output producer.
FILAMENT_BRIDGE_API int filament_bridge_set_screen_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore);
FILAMENT_BRIDGE_API int filament_bridge_apply_animations(
        FilamentBridge* bridge, double time_seconds);
FILAMENT_BRIDGE_API uint32_t filament_bridge_animation_count(
        const FilamentBridge* bridge);
FILAMENT_BRIDGE_API float filament_bridge_animation_duration(
        const FilamentBridge* bridge, uint32_t animation_index);
FILAMENT_BRIDGE_API const char* filament_bridge_last_error(
        const FilamentBridge* bridge);

FILAMENT_BRIDGE_API FilamentPreview* filament_preview_create(
        void* native_window, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API void filament_preview_destroy(FilamentPreview* preview);
FILAMENT_BRIDGE_API int filament_preview_load_glb(
        FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_preview_apply_animations(
        FilamentPreview* preview, double time_seconds);
FILAMENT_BRIDGE_API int filament_preview_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z);
FILAMENT_BRIDGE_API int filament_preview_set_projection(
        FilamentPreview* preview,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_preview_set_viewport(
        FilamentPreview* preview, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_preview_set_scene_exposure(
        FilamentPreview* preview, float exposure_ev);
FILAMENT_BRIDGE_API int filament_preview_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z);
FILAMENT_BRIDGE_API int filament_preview_set_skybox_brightness(
        FilamentPreview* preview, float brightness);
FILAMENT_BRIDGE_API int filament_preview_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees);
FILAMENT_BRIDGE_API int filament_preview_render(FilamentPreview* preview);
FILAMENT_BRIDGE_API const char* filament_preview_last_error(
        const FilamentPreview* preview);

}
