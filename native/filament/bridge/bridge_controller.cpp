#include "bridge_controller.h"
#include "bridge_internal.h"

void bridge_controller_destroy(FilamentBridge* bridge, ControllerAsset& controller) {
    if (controller.asset && bridge->scene) {
        bridge->scene->removeEntities(
                controller.asset->getEntities(), controller.asset->getEntityCount());
    }
    if (controller.asset && bridge->asset_loader) {
        bridge->asset_loader->destroyAsset(controller.asset);
    }
    controller = {};
}

std::string bridge_controller_semantic(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (name.find("thumbstick_xaxis") != std::string::npos ||
            name.find("touchpad_xaxis") != std::string::npos) return "joystick_x";
    if (name.find("thumbstick_yaxis") != std::string::npos ||
            name.find("touchpad_yaxis") != std::string::npos) return "joystick_y";
    if (name.find("thumbstick") != std::string::npos ||
            name.find("touchpad") != std::string::npos) return "joystick";
    if (name.find("trigger") != std::string::npos) return "trigger";
    if (name.find("squeeze") != std::string::npos ||
            name.find("grip") != std::string::npos ||
            name.find("grasp") != std::string::npos) return "grip";
    if (name.find("a_button") != std::string::npos ||
            name.find("abutton") != std::string::npos) return "a_button";
    if (name.find("b_button") != std::string::npos ||
            name.find("bbutton") != std::string::npos) return "b_button";
    if (name.find("x_button") != std::string::npos ||
            name.find("xbutton") != std::string::npos) return "x_button";
    if (name.find("y_button") != std::string::npos ||
            name.find("ybutton") != std::string::npos) return "y_button";
    if (name.find("menu") != std::string::npos) return "menu_button";
    if (name.find("photo_button") != std::string::npos ||
            name.find("home_button") != std::string::npos ||
            name.find("app_button") != std::string::npos ||
            name.find("pico") != std::string::npos) return "menu_button";
    return {};
}

filament::math::mat4f bridge_controller_interpolate_transform(
        const filament::math::mat4f& value,
        const filament::math::mat4f& minimum,
        const filament::math::mat4f& maximum,
        float amount) {
    const float t = std::clamp(std::abs(amount), 0.0f, 1.0f);
    const auto& target = amount < 0.0f ? minimum : maximum;
    filament::math::mat4f result = value;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result[column][row] = value[column][row] +
                    (target[column][row] - value[column][row]) * t;
        }
    }
    return result;
}

float bridge_controller_animation_amount(
        const ControllerAsset& controller, const std::string& semantic) {
    if (semantic == "trigger") return controller.trigger;
    if (semantic == "grip") return controller.grip;
    if (semantic == "joystick_x") return controller.joystick_x;
    if (semantic == "joystick_y") return controller.joystick_y;
    if (semantic == "joystick") return controller.joystick_x != 0.0f ||
            controller.joystick_y != 0.0f ||
            (controller.button_mask & (1u << 5)) ? 1.0f : 0.0f;
    if (semantic == "a_button") return (controller.button_mask & (1u << 0)) ? 1.0f : 0.0f;
    if (semantic == "b_button") return (controller.button_mask & (1u << 1)) ? 1.0f : 0.0f;
    if (semantic == "x_button") return (controller.button_mask & (1u << 2)) ? 1.0f : 0.0f;
    if (semantic == "y_button") return (controller.button_mask & (1u << 3)) ? 1.0f : 0.0f;
    if (semantic == "menu_button") return (controller.button_mask & (1u << 4)) ? 1.0f : 0.0f;
    return 0.0f;
}

void bridge_controller_update_animations(
        FilamentBridge* bridge, ControllerAsset& controller) {
    if (!controller.asset || !bridge->engine) return;
    auto& transforms = bridge->engine->getTransformManager();
    for (const auto& animation : controller.animations) {
        const float amount = bridge_controller_animation_amount(controller, animation.semantic);
        if (!transforms.hasComponent(animation.value_entity)) continue;
        transforms.setTransform(
                transforms.getInstance(animation.value_entity),
                bridge_controller_interpolate_transform(
                        animation.value_transform,
                        animation.min_transform,
                        animation.max_transform,
                        amount));
    }
}

int bridge_controller_load(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count) {
    if (!bridge || !bridge->engine || !bridge->asset_loader ||
            hand > 1 || !bytes || !byte_count) {
        return 0;
    }
    auto& controller = bridge->controllers[hand];
    bridge_controller_destroy(bridge, controller);
    controller.bytes.assign(bytes, bytes + byte_count);
    controller.asset = bridge->asset_loader->createAsset(
            controller.bytes.data(), byte_count);
    if (!controller.asset) {
        bridge_set_error(bridge, "Filament could not parse controller GLB");
        controller = {};
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{bridge->engine, nullptr, true};
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", bridge->texture_provider);
    resources.addTextureProvider("image/jpeg", bridge->texture_provider);
    if (!resources.loadResources(controller.asset)) {
        bridge_controller_destroy(bridge, controller);
        bridge_set_error(bridge, "Filament could not load controller GLB resources");
        return 0;
    }
    bridge->scene->addEntities(
            controller.asset->getEntities(), controller.asset->getEntityCount());
    // Controllers are runtime overlays lit by the shared fill-light channel.
    // Without this channel assignment their PBR materials receive no light
    // because the environment asset owns the default channel routing.
    auto& renderables = bridge->engine->getRenderableManager();
    for (size_t index = 0; index < controller.asset->getRenderableEntityCount(); ++index) {
        const auto entity = controller.asset->getRenderableEntities()[index];
        const auto instance = renderables.getInstance(entity);
        if (!instance.isValid()) continue;
        renderables.setLightChannel(instance, 1, true);
    }
    const auto& transforms = bridge->engine->getTransformManager();
    for (size_t index = 0; index < controller.asset->getEntityCount(); ++index) {
        const auto entity = controller.asset->getEntities()[index];
        const char* raw_name = controller.asset->getName(entity);
        if (!raw_name) continue;
        const std::string value_name(raw_name);
        const std::string suffix = "_value";
        if (value_name.size() <= suffix.size() ||
                value_name.compare(value_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const std::string prefix = value_name.substr(0, value_name.size() - suffix.size());
        const auto min_entity = controller.asset->getFirstEntityByName(
                (prefix + "_min").c_str());
        const auto max_entity = controller.asset->getFirstEntityByName(
                (prefix + "_max").c_str());
        const auto value_instance = transforms.getInstance(entity);
        const auto min_instance = transforms.getInstance(min_entity);
        const auto max_instance = transforms.getInstance(max_entity);
        if (min_entity.isNull() || max_entity.isNull() ||
                !value_instance.isValid() || !min_instance.isValid() || !max_instance.isValid()) {
            continue;
        }
        const std::string semantic = bridge_controller_semantic(value_name);
        if (semantic.empty()) continue;
        controller.animations.push_back({
                entity, min_entity, max_entity,
                transforms.getTransform(value_instance),
                transforms.getTransform(min_instance),
                transforms.getTransform(max_instance),
                semantic});
    }
    controller.asset->releaseSourceData();
    controller.bytes.clear();
    bridge->engine->flushAndWait();
    bridge_controller_update_animations(bridge, controller);
    return 1;
}

int bridge_controller_set_pose(
        FilamentBridge* bridge, uint32_t hand, const float* matrix16) {
    if (!bridge || !bridge->engine || hand > 1 || !matrix16 ||
            !bridge->controllers[hand].asset) return 0;
    const auto root = bridge->controllers[hand].asset->getRoot();
    auto& transforms = bridge->engine->getTransformManager();
    const auto instance = transforms.getInstance(root);
    if (!instance.isValid()) return 0;
    const filament::math::mat4f matrix(
            matrix16[0], matrix16[1], matrix16[2], matrix16[3],
            matrix16[4], matrix16[5], matrix16[6], matrix16[7],
            matrix16[8], matrix16[9], matrix16[10], matrix16[11],
            matrix16[12], matrix16[13], matrix16[14], matrix16[15]);
    transforms.setTransform(instance, matrix);
    return 1;
}

int bridge_controller_set_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip,
        float joystick_x, float joystick_y,
        uint32_t button_mask) {
    if (!bridge || hand > 1 || !bridge->controllers[hand].asset) return 0;
    auto& controller = bridge->controllers[hand];
    controller.trigger = std::clamp(trigger, 0.0f, 1.0f);
    controller.grip = std::clamp(grip, 0.0f, 1.0f);
    controller.joystick_x = std::clamp(joystick_x, -1.0f, 1.0f);
    controller.joystick_y = std::clamp(joystick_y, -1.0f, 1.0f);
    controller.button_mask = button_mask;
    bridge_controller_update_animations(bridge, controller);
    return 1;
}

int bridge_controller_set_visible(
        FilamentBridge* bridge, uint32_t hand, int visible) {
    if (!bridge || !bridge->engine || hand > 1 ||
            !bridge->controllers[hand].asset) return 0;
    auto& controller = bridge->controllers[hand];
    const bool next_visible = visible != 0;
    if (controller.visible == next_visible) return 1;
    for (size_t index = 0;
            index < controller.asset->getRenderableEntityCount(); ++index) {
        const auto entity = controller.asset->getRenderableEntities()[index];
        bridge_set_renderable_visible(bridge, entity, next_visible);
    }
    controller.visible = next_visible;
    return 1;
}
