#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_eye_activate(FilamentBridge* bridge, uint32_t eye_index);
int bridge_eye_create_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height);
int bridge_eye_create_target_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height);
int bridge_eye_set_active(FilamentBridge* bridge, uint32_t eye_index);
int bridge_eye_set_acquired_image(FilamentBridge* bridge, uint32_t image_index);
int bridge_eye_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z);
int bridge_eye_set_camera_projection(
        FilamentBridge* bridge, double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane);
int bridge_eye_set_camera_projection_frustum(
        FilamentBridge* bridge, double left, double right,
        double bottom, double top, double near_plane, double far_plane);
int bridge_eye_begin_frame(FilamentBridge* bridge);
int bridge_eye_end_frame(FilamentBridge* bridge);
int bridge_eye_set_ready_semaphore(FilamentBridge* bridge, const void* semaphore);
