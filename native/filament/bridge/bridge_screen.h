#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_screen_destroy(FilamentBridge* bridge);
int bridge_screen_create(FilamentBridge* bridge);
int bridge_screen_update(
        FilamentBridge* bridge,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees,
        float rotation_z_degrees);
int bridge_screen_set_curved(FilamentBridge* bridge, int curved);
int bridge_screen_set_light(
        FilamentBridge* bridge,
        float red, float green, float blue, float intensity);
int bridge_screen_set_sampling(FilamentBridge* bridge, float filter_scale);
int bridge_screen_set_sampling_mode(FilamentBridge* bridge, int use_mip);
int bridge_screen_set_image(
        FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format);
int bridge_screen_set_source_version(
        FilamentBridge* bridge, uint64_t version);
int bridge_screen_set_fixed_image(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height);
int bridge_screen_prepare_frame(FilamentBridge* bridge);
int bridge_screen_get_sampling_stats(
        FilamentBridge* bridge, uint32_t eye_index,
        uint64_t* source_binds, uint64_t* mip_generations);
