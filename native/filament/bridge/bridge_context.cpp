#include "bridge_context.h"
#include "bridge_internal.h"
#include "bridge_controller.h"
#include "bridge_eye.h"
#include "bridge_laser.h"
#include "bridge_material.h"
#include "bridge_scene.h"
#include "bridge_screen.h"

void bridge_set_error(FilamentBridge* bridge, const char* message) {
    if (bridge) {
        bridge->last_error = message;
    }
}

void bridge_set_renderable_visible(
        FilamentBridge* bridge, utils::Entity entity, bool visible) {
    bridge_set_renderable_layer(bridge, entity, 0, visible);
}

void bridge_set_renderable_layer(
        FilamentBridge* bridge, utils::Entity entity,
        uint8_t layer, bool visible) {
    if (!bridge || !bridge->engine || entity.isNull()) return;
    auto& renderables = bridge->engine->getRenderableManager();
    const auto instance = renderables.getInstance(entity);
    if (!instance.isValid()) return;
    const uint8_t layer_mask = static_cast<uint8_t>(1u << layer);
    renderables.setLayerMask(instance, 0xff, visible ? layer_mask : 0x00);
}

FilamentBridge* bridge_context_create(
        const FilamentBridgeVulkanCreateInfo* info) {
    auto bridge = std::make_unique<FilamentBridge>();
    if (!info || !info->instance || !info->physical_device || !info->device) {
        bridge_set_error(bridge.get(), "Vulkan create info contains a null handle");
        return bridge.release();
    }

    bridge->shared_context.instance = reinterpret_cast<VkInstance>(info->instance);
    bridge->shared_context.physicalDevice = reinterpret_cast<VkPhysicalDevice>(
            info->physical_device);
    bridge->shared_context.logicalDevice = reinterpret_cast<VkDevice>(info->device);
    bridge->shared_context.graphicsQueueFamilyIndex =
            info->graphics_queue_family_index;
    bridge->shared_context.graphicsQueueIndex = info->graphics_queue_index;
    bridge->platform = new OpenXrVulkanPlatform();
    bridge->engine = filament::Engine::Builder()
            .backend(filament::Engine::Backend::VULKAN)
            .platform(bridge->platform)
            .sharedContext(&bridge->shared_context)
            .build();
    if (!bridge->engine) {
        bridge_set_error(bridge.get(), "Filament Vulkan Engine creation failed");
        delete bridge->platform;
        bridge->platform = nullptr;
        return bridge.release();
    }
    bridge->scene = bridge->engine->createScene();
    bridge->materials = filament::gltfio::createJitShaderProvider(bridge->engine);
    bridge->texture_provider = filament::gltfio::createStbProvider(bridge->engine);
    if (!bridge->scene || !bridge->materials ||
            !bridge->texture_provider) {
        bridge_set_error(bridge.get(), "Filament Vulkan resource creation failed");
        return bridge.release();
    }
    for (auto& eye : bridge->eyes) {
        eye.renderer = bridge->engine->createRenderer();
        eye.view = bridge->engine->createView();
        eye.laser_view = bridge->engine->createView();
        eye.camera = bridge->engine->createCamera(
                utils::EntityManager::get().create());
        if (!eye.renderer || !eye.view || !eye.laser_view || !eye.camera) {
            bridge_set_error(bridge.get(), "Filament Vulkan eye resource creation failed");
            return bridge.release();
        }
        eye.camera->lookAt(
                filament::math::float3{0.0f, 0.0f, 3.0f},
                filament::math::float3{0.0f, 0.0f, 0.0f},
                filament::math::float3{0.0f, 1.0f, 0.0f});
        eye.view->setScene(bridge->scene);
        eye.view->setCamera(eye.camera);
        eye.view->setVisibleLayers(0xff, 0x01);
        eye.view->setChannelDepthClearEnabled(0, false);
        eye.laser_view->setScene(bridge->scene);
        eye.laser_view->setCamera(eye.camera);
        // Layer 1 contains display-referred screen/UI geometry and lasers.
        eye.laser_view->setVisibleLayers(0xff, 0x02);
        eye.laser_view->setColorGrading(nullptr);
        eye.laser_view->setPostProcessingEnabled(false);
        eye.laser_view->setChannelDepthClearEnabled(0, false);
    }
    bridge_eye_activate(bridge.get(), 0);
    for (auto& eye : bridge->eyes) {
        bridge->view = eye.view;
        bridge->camera = eye.camera;
        bridge->color_grading = nullptr;
        if (!bridge_material_configure_color_pipeline(bridge.get())) {
            bridge_set_error(bridge.get(), "Filament Vulkan color pipeline creation failed");
            return bridge.release();
        }
        eye.color_grading = bridge->color_grading;
    }
    bridge_eye_activate(bridge.get(), 0);
    filament::gltfio::AssetConfiguration config{bridge->engine, bridge->materials};
    bridge->asset_loader = filament::gltfio::AssetLoader::create(config);
    if (!bridge->asset_loader) {
        bridge_set_error(bridge.get(), "Filament AssetLoader creation failed");
    } else if (!bridge_laser_create(bridge.get())) {
        return bridge.release();
    }
    return bridge.release();
}

void bridge_context_destroy(FilamentBridge* bridge) {
    if (!bridge) return;
    bridge_scene_destroy(bridge);
    for (auto& controller : bridge->controllers) {
        bridge_controller_destroy(bridge, controller);
    }
    bridge_laser_destroy(bridge);
    bridge_screen_destroy(bridge);
    for (auto& eye : bridge->eyes) {
        if (eye.renderer && bridge->engine) {
            bridge->engine->destroy(eye.renderer);
        }
        if (eye.swapchain && bridge->engine) {
            bridge->engine->destroy(eye.swapchain);
        }
        if (eye.color_grading && bridge->engine) {
            if (eye.view) eye.view->setColorGrading(nullptr);
            bridge->engine->destroy(eye.color_grading);
        }
        if (eye.laser_view && bridge->engine) {
            bridge->engine->destroy(eye.laser_view);
        }
        if (eye.view && bridge->engine) {
            bridge->engine->destroy(eye.view);
        }
        if (eye.camera && bridge->engine) {
            bridge->engine->destroy(eye.camera->getEntity());
        }
    }
    if (!bridge->fill_light.isNull() && bridge->engine) {
        bridge->engine->destroy(bridge->fill_light);
    }
    if (bridge->scene && bridge->engine) {
        bridge->engine->destroy(bridge->scene);
    }
    if (bridge->asset_loader) {
        filament::gltfio::AssetLoader::destroy(&bridge->asset_loader);
    }
    if (bridge->materials) {
        bridge->materials->destroyMaterials();
        delete bridge->materials;
    }
    delete bridge->texture_provider;
    if (bridge->engine) {
        filament::Engine::destroy(&bridge->engine);
    }
    delete bridge->platform;
    delete bridge;
}

int bridge_context_wait_for_idle(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return 0;
    bridge->engine->flushAndWait();
    return 1;
}

const char* bridge_context_last_error(const FilamentBridge* bridge) {
    return bridge ? bridge->last_error.c_str() : "bridge is null";
}
