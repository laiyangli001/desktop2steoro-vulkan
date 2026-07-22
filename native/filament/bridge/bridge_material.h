#pragma once

struct FilamentBridge;
struct FilamentPreview;

bool bridge_material_configure_color_pipeline(FilamentBridge* bridge);
bool bridge_material_configure_color_pipeline(FilamentPreview* preview);
void bridge_material_collect_brightness(
        FilamentBridge* bridge, bool enable_fill_channel);
void bridge_material_collect_brightness(
        FilamentPreview* preview, bool enable_fill_channel);
void bridge_material_apply_brightness(FilamentBridge* bridge);
void bridge_material_apply_brightness(FilamentPreview* preview);
int bridge_material_set_scene_exposure(
        FilamentBridge* bridge, float exposure_ev);
int bridge_material_set_skybox_brightness(
        FilamentBridge* bridge, float brightness);
int bridge_material_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue, float intensity,
        float direction_x, float direction_y, float direction_z);
int preview_material_set_scene_exposure(
        FilamentPreview* preview, float exposure_ev);
int preview_material_set_skybox_brightness(
        FilamentPreview* preview, float brightness);
int preview_material_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue, float intensity,
        float direction_x, float direction_y, float direction_z);
