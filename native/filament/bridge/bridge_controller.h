#pragma once

#include <cstdint>

struct ControllerAsset;
struct FilamentBridge;

int bridge_controller_create_occlusion_material(FilamentBridge* bridge);
void bridge_controller_destroy_occlusion_material(FilamentBridge* bridge);
void bridge_controller_set_occlusion_materials(
        FilamentBridge* bridge, bool enabled);
void bridge_controller_destroy(
        FilamentBridge* bridge, ControllerAsset& controller);
void bridge_controller_update_animations(
        FilamentBridge* bridge, ControllerAsset& controller);
int bridge_controller_load(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count);
int bridge_controller_set_pose(
        FilamentBridge* bridge, uint32_t hand, const float* matrix16);
int bridge_controller_set_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip, float joystick_x, float joystick_y,
        uint32_t button_mask);
int bridge_controller_set_visible(
        FilamentBridge* bridge, uint32_t hand, int visible);
