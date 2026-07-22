#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_laser_destroy(FilamentBridge* bridge);
int bridge_laser_create(FilamentBridge* bridge);
int bridge_laser_set(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16, int visible);
