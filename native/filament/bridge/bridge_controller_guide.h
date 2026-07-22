#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_controller_guide_destroy(FilamentBridge* bridge);
int bridge_controller_guide_set_texture(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height);
int bridge_controller_guide_set(
        FilamentBridge* bridge, const float* matrix16, int visible);
