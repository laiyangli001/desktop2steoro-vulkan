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
int bridge_screen_set_image(
        FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format);
