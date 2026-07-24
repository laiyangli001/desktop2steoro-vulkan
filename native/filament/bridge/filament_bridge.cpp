#include "filament_bridge.h"

#include "bridge_context.h"
#include "bridge_controller.h"
#include "bridge_controller_guide.h"
#include "bridge_eye.h"
#include "bridge_laser.h"
#include "bridge_material.h"
#include "bridge_scene.h"
#include "bridge_screen.h"
#include "preview_bridge.h"

FilamentBridge* filament_bridge_create_vulkan(
        const FilamentBridgeVulkanCreateInfo* info) {
    return bridge_context_create(info);
}

void filament_bridge_destroy(FilamentBridge* bridge) {
    bridge_context_destroy(bridge);
}

int filament_bridge_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height) {
    return bridge_eye_create_swapchain(
            bridge, image_handles, image_count, format, width, height);
}

int filament_bridge_create_eye_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height) {
    return bridge_eye_create_target_swapchain(
            bridge, eye_index, image_handles, image_count, format, width, height);
}

int filament_bridge_set_active_eye(
        FilamentBridge* bridge, uint32_t eye_index) {
    return bridge_eye_set_active(bridge, eye_index);
}

int filament_bridge_set_acquired_image(
        FilamentBridge* bridge, uint32_t image_index) {
    return bridge_eye_set_acquired_image(bridge, image_index);
}

int filament_bridge_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    return bridge_eye_set_camera_look_at(
            bridge, eye_x, eye_y, eye_z,
            center_x, center_y, center_z, up_x, up_y, up_z);
}

int filament_bridge_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    return bridge_eye_set_camera_projection(
            bridge, vertical_fov_degrees, aspect, near_plane, far_plane);
}

int filament_bridge_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane) {
    return bridge_eye_set_camera_projection_frustum(
            bridge, left, right, bottom, top, near_plane, far_plane);
}

int filament_bridge_begin_frame(FilamentBridge* bridge) {
    return bridge_eye_begin_frame(bridge);
}

int filament_bridge_end_frame(FilamentBridge* bridge) {
    return bridge_eye_end_frame(bridge);
}

int filament_bridge_wait_for_idle(FilamentBridge* bridge) {
    return bridge_context_wait_for_idle(bridge);
}

int filament_bridge_load_glb(
        FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count) {
    return bridge_scene_load_glb(bridge, bytes, byte_count);
}

int filament_bridge_load_controller(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count) {
    return bridge_controller_load(bridge, hand, bytes, byte_count);
}

int filament_bridge_set_controller_pose(
        FilamentBridge* bridge, uint32_t hand, const float* matrix16) {
    return bridge_controller_set_pose(bridge, hand, matrix16);
}

int filament_bridge_set_controller_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip,
        float joystick_x, float joystick_y,
        uint32_t button_mask) {
    return bridge_controller_set_inputs(
            bridge, hand, trigger, grip, joystick_x, joystick_y, button_mask);
}

int filament_bridge_set_controller_visible(
        FilamentBridge* bridge, uint32_t hand, int visible) {
    return bridge_controller_set_visible(bridge, hand, visible);
}

int filament_bridge_set_controller_laser(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16, int visible) {
    return bridge_laser_set(bridge, hand, matrix16, visible);
}

int filament_bridge_set_controller_guide_texture(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height) {
    return bridge_controller_guide_set_texture(bridge, rgba, width, height);
}

int filament_bridge_set_controller_guide(
        FilamentBridge* bridge, const float* matrix16, int visible) {
    return bridge_controller_guide_set(bridge, matrix16, visible);
}

int filament_bridge_set_scene_exposure(
        FilamentBridge* bridge, float exposure_ev) {
    return bridge_material_set_scene_exposure(bridge, exposure_ev);
}

int filament_bridge_set_skybox_brightness(
        FilamentBridge* bridge, float brightness) {
    return bridge_material_set_skybox_brightness(bridge, brightness);
}

int filament_bridge_set_passthrough_backdrop(
        FilamentBridge* bridge, int enabled) {
    return bridge_material_set_passthrough_backdrop(bridge, enabled);
}

int filament_bridge_set_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue) {
    return bridge_material_set_ambient_light(bridge, red, green, blue);
}

int filament_bridge_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    return bridge_material_set_fill_light(
            bridge, red, green, blue, intensity,
            direction_x, direction_y, direction_z);
}

int filament_bridge_create_screen(FilamentBridge* bridge) {
    return bridge_screen_create(bridge);
}

int filament_bridge_set_screen(
        FilamentBridge* bridge,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees,
        float rotation_z_degrees) {
    return bridge_screen_update(
            bridge, position_x, position_y, position_z, width, height,
            rotation_x_degrees, rotation_y_degrees, rotation_z_degrees);
}

int filament_bridge_set_screen_curved(FilamentBridge* bridge, int curved) {
    return bridge_screen_set_curved(bridge, curved);
}

int filament_bridge_set_screen_light(
        FilamentBridge* bridge,
        float red, float green, float blue, float intensity) {
    return bridge_screen_set_light(bridge, red, green, blue, intensity);
}

int filament_bridge_set_screen_image(
        FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format) {
    return bridge_screen_set_image(bridge, image, width, height, format);
}

int filament_bridge_set_screen_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore) {
    return bridge_eye_set_ready_semaphore(bridge, semaphore);
}

int filament_bridge_get_finished_drawing_semaphore(
        FilamentBridge* bridge, const void** semaphore) {
    return bridge_eye_get_finished_semaphore(bridge, semaphore);
}

int filament_bridge_apply_animations(
        FilamentBridge* bridge, double time_seconds) {
    return bridge_scene_apply_animations(bridge, time_seconds);
}

uint32_t filament_bridge_animation_count(const FilamentBridge* bridge) {
    return bridge_scene_animation_count(bridge);
}

float filament_bridge_animation_duration(
        const FilamentBridge* bridge, uint32_t animation_index) {
    return bridge_scene_animation_duration(bridge, animation_index);
}

const char* filament_bridge_last_error(const FilamentBridge* bridge) {
    return bridge_context_last_error(bridge);
}

FilamentPreview* filament_preview_create(
        void* native_window, uint32_t width, uint32_t height) {
    return preview_bridge_create(native_window, width, height);
}

void filament_preview_destroy(FilamentPreview* preview) {
    preview_bridge_destroy(preview);
}

int filament_preview_load_glb(
        FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count) {
    return preview_bridge_load_glb(preview, bytes, byte_count);
}

int filament_preview_apply_animations(
        FilamentPreview* preview, double time_seconds) {
    return preview_bridge_apply_animations(preview, time_seconds);
}

int filament_preview_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    return preview_bridge_set_camera(
            preview, eye_x, eye_y, eye_z,
            center_x, center_y, center_z, up_x, up_y, up_z);
}

int filament_preview_set_projection(
        FilamentPreview* preview,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    return preview_bridge_set_projection(
            preview, vertical_fov_degrees, aspect, near_plane, far_plane);
}

int filament_preview_set_viewport(
        FilamentPreview* preview, uint32_t width, uint32_t height) {
    return preview_bridge_set_viewport(preview, width, height);
}

int filament_preview_set_scene_exposure(
        FilamentPreview* preview, float exposure_ev) {
    return preview_material_set_scene_exposure(preview, exposure_ev);
}

int filament_preview_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    return preview_material_set_fill_light(
            preview, red, green, blue, intensity,
            direction_x, direction_y, direction_z);
}

int filament_preview_set_skybox_brightness(
        FilamentPreview* preview, float brightness) {
    return preview_material_set_skybox_brightness(preview, brightness);
}

int filament_preview_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees,
        float rotation_z_degrees) {
    return preview_bridge_set_screen(
            preview, position_x, position_y, position_z, width, height,
            rotation_x_degrees, rotation_y_degrees, rotation_z_degrees);
}

int filament_preview_render(FilamentPreview* preview) {
    return preview_bridge_render(preview);
}

const char* filament_preview_last_error(const FilamentPreview* preview) {
    return preview_bridge_last_error(preview);
}
