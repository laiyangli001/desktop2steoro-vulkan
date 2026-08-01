#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_glow_destroy(FilamentBridge* bridge);
int bridge_glow_create(FilamentBridge* bridge);
int bridge_glow_set_source(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height);
int bridge_glow_set_image(
        FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format);
int bridge_glow_set_state(
        FilamentBridge* bridge, uint32_t mode,
        float head_x, float head_y, float head_z,
        float glow_intensity, float glow_width,
        float glow_intensity_multiplier,
        float frosted_intensity, float frosted_alpha,
        float frosted_threshold, float frosted_lod,
        float frosted_blend, float frosted_thickness,
        float frosted_diffuse, float frosted_inset,
        float veil_intensity, float veil_alpha);
void bridge_glow_update_geometry(FilamentBridge* bridge);
void bridge_glow_update_visibility(FilamentBridge* bridge);
