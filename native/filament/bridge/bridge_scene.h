#pragma once

#include <cstdint>

struct FilamentBridge;

void bridge_scene_destroy(FilamentBridge* bridge);
int bridge_scene_load_glb(
        FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count);
int bridge_scene_apply_animations(FilamentBridge* bridge, double time_seconds);
uint32_t bridge_scene_animation_count(const FilamentBridge* bridge);
float bridge_scene_animation_duration(
        const FilamentBridge* bridge, uint32_t animation_index);
