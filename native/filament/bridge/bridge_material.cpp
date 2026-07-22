#include "bridge_material.h"
#include "bridge_internal.h"

template<typename Target>
bool configure_color_pipeline_impl(Target* target) {
    if (!target || !target->engine || !target->view) {
        return false;
    }
    target->color_grading = filament::ColorGrading::Builder()
            .toneMapping(filament::ColorGrading::ToneMapping::ACES_LEGACY)
            // Keep the projection target in sRGB format and let its target
            // conversion perform the single sRGB OETF at store time.
            .outputColorSpace(filament::color::Rec709 - filament::color::Linear - filament::color::D65)
            .build(*target->engine);
    if (!target->color_grading) {
        return false;
    }
    target->view->setColorGrading(target->color_grading);
    target->view->setPostProcessingEnabled(true);
    return true;
}

bool is_skybox_name(const char* name) {
    if (!name) return false;
    std::string value(name);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value.find("skybox") != std::string::npos;
}

template<typename BridgeType>
void apply_material_brightness_impl(BridgeType* bridge) {
    if (!bridge || !bridge->engine) return;
    const float scene_factor = std::exp2(bridge->brightness.scene_exposure_ev);
    for (const auto& entry : bridge->brightness.scene_materials) {
        if (!entry.material) continue;
        const auto& base = entry.base_color_factor;
        entry.material->setParameter("baseColorFactor", filament::math::float4{
                base.x * scene_factor, base.y * scene_factor,
                base.z * scene_factor, base.w});
    }
    const float skybox_factor = bridge->brightness.skybox_brightness;
    for (const auto& entry : bridge->brightness.skybox_materials) {
        if (!entry.material) continue;
        const auto& base = entry.base_color_factor;
        entry.material->setParameter("baseColorFactor", filament::math::float4{
                base.x * skybox_factor, base.y * skybox_factor,
                base.z * skybox_factor, base.w});
    }
}

template<typename BridgeType>
void collect_material_brightness_impl(BridgeType* bridge, bool enable_fill_channel) {
    if (!bridge || !bridge->engine || !bridge->asset) return;
    auto& renderables = bridge->engine->getRenderableManager();
    bridge->brightness.scene_materials.clear();
    bridge->brightness.skybox_materials.clear();
    const auto* entities = bridge->asset->getRenderableEntities();
    for (size_t index = 0; index < bridge->asset->getRenderableEntityCount(); ++index) {
        const utils::Entity entity = entities[index];
        auto instance = renderables.getInstance(entity);
        if (!instance.isValid()) continue;
        const bool skybox = is_skybox_name(bridge->asset->getName(entity));
        if (skybox) {
            // Render the exported skybox before regular scene geometry so its
            // depth buffer cannot hide the Saturn ring or other scene meshes.
            renderables.setPriority(instance, 0);
            renderables.setLightChannel(instance, 0, false);
            renderables.setLightChannel(instance, 1, false);
        } else if (enable_fill_channel) {
            renderables.setLightChannel(instance, 1, true);
        }
        auto& target = skybox
                ? bridge->brightness.skybox_materials
                : bridge->brightness.scene_materials;
        for (size_t primitive = 0; primitive < renderables.getPrimitiveCount(instance); ++primitive) {
            auto* material = renderables.getMaterialInstanceAt(instance, primitive);
            if (!material || !material->getMaterial()->hasParameter("baseColorFactor")) continue;
            target.push_back({material, material->template getParameter<filament::math::float4>(
                    "baseColorFactor")});
        }
    }
    apply_material_brightness_impl(bridge);
}

int preview_material_set_scene_exposure(FilamentPreview* preview, float exposure_ev) {
    if (!preview || !preview->engine || !std::isfinite(exposure_ev)) {
        return 0;
    }
    preview->brightness.scene_exposure_ev = std::clamp(exposure_ev, -8.0f, 8.0f);
    bridge_material_apply_brightness(preview);
    return 1;
}

int preview_material_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    if (!preview || !preview->engine || !preview->scene ||
            !std::isfinite(red) || !std::isfinite(green) || !std::isfinite(blue) ||
            !std::isfinite(intensity) || intensity < 0.0f ||
            !std::isfinite(direction_x) || !std::isfinite(direction_y) ||
            !std::isfinite(direction_z)) {
        return 0;
    }
    if (!preview->fill_light.isNull()) {
        preview->scene->remove(preview->fill_light);
        preview->engine->destroy(preview->fill_light);
    }
    preview->fill_light = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
            .color(filament::LinearColor{red, green, blue})
            .intensity(intensity)
            .direction({direction_x, direction_y, direction_z})
            .lightChannel(0, false)
            .lightChannel(1, true)
            .castShadows(false)
            .build(*preview->engine, preview->fill_light);
    preview->scene->addEntity(preview->fill_light);
    return 1;
}

int preview_material_set_skybox_brightness(FilamentPreview* preview, float brightness) {
    if (!preview || !std::isfinite(brightness) || brightness < 0.0f) return 0;
    preview->brightness.skybox_brightness = std::min(brightness, 16.0f);
    bridge_material_apply_brightness(preview);
    return 1;
}

int bridge_material_set_scene_exposure(FilamentBridge* bridge, float exposure_ev) {
    if (!bridge || !std::isfinite(exposure_ev)) return 0;
    bridge->brightness.scene_exposure_ev = std::clamp(exposure_ev, -8.0f, 8.0f);
    apply_material_brightness_impl(bridge);
    return 1;
}

int bridge_material_set_skybox_brightness(FilamentBridge* bridge, float brightness) {
    if (!bridge || !std::isfinite(brightness) || brightness < 0.0f) return 0;
    bridge->brightness.skybox_brightness = std::min(brightness, 16.0f);
    apply_material_brightness_impl(bridge);
    return 1;
}

int bridge_material_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    if (!bridge || !bridge->engine || !bridge->scene ||
            !std::isfinite(red) || !std::isfinite(green) ||
            !std::isfinite(blue) || !std::isfinite(intensity) || intensity < 0.0f ||
            !std::isfinite(direction_x) || !std::isfinite(direction_y) ||
            !std::isfinite(direction_z)) {
        return 0;
    }
    if (!bridge->fill_light.isNull()) {
        bridge->scene->remove(bridge->fill_light);
        bridge->engine->destroy(bridge->fill_light);
        bridge->fill_light = {};
    }
    bridge->fill_light = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
            .color(filament::LinearColor{red, green, blue})
            .intensity(intensity)
            .direction({direction_x, direction_y, direction_z})
            .lightChannel(0, false)
            .lightChannel(1, true)
            .castShadows(false)
            .build(*bridge->engine, bridge->fill_light);
    bridge->scene->addEntity(bridge->fill_light);
    return 1;
}

bool bridge_material_configure_color_pipeline(FilamentBridge* bridge) {
    return configure_color_pipeline_impl(bridge);
}

bool bridge_material_configure_color_pipeline(FilamentPreview* preview) {
    return configure_color_pipeline_impl(preview);
}

void bridge_material_collect_brightness(
        FilamentBridge* bridge, bool enable_fill_channel) {
    collect_material_brightness_impl(bridge, enable_fill_channel);
}

void bridge_material_collect_brightness(
        FilamentPreview* preview, bool enable_fill_channel) {
    collect_material_brightness_impl(preview, enable_fill_channel);
}

void bridge_material_apply_brightness(FilamentBridge* bridge) {
    apply_material_brightness_impl(bridge);
}

void bridge_material_apply_brightness(FilamentPreview* preview) {
    apply_material_brightness_impl(preview);
}
