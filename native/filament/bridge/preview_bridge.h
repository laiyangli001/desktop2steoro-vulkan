#pragma once

#include <cstdint>

struct FilamentPreview;

FilamentPreview* preview_bridge_create(
        void* native_window, uint32_t width, uint32_t height);
void preview_bridge_destroy(FilamentPreview* preview);
int preview_bridge_load_glb(
        FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count);
int preview_bridge_apply_animations(
        FilamentPreview* preview, double time_seconds);
int preview_bridge_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z);
int preview_bridge_set_projection(
        FilamentPreview* preview, double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane);
int preview_bridge_set_viewport(
        FilamentPreview* preview, uint32_t width, uint32_t height);
int preview_bridge_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees,
        float rotation_z_degrees);
int preview_bridge_render(FilamentPreview* preview);
const char* preview_bridge_last_error(const FilamentPreview* preview);
