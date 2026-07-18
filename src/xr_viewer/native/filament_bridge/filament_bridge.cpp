#include "filament_bridge.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <filament/Engine.h>
#include <gltfio/Animator.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/TextureProvider.h>

struct FilamentBridge {
    filament::Engine* engine = nullptr;
    filament::gltfio::MaterialProvider* materials = nullptr;
    filament::gltfio::TextureProvider* texture_provider = nullptr;
    filament::gltfio::AssetLoader* asset_loader = nullptr;
    filament::gltfio::FilamentAsset* asset = nullptr;
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
};

namespace {

void destroy_asset(FilamentBridge* bridge) {
    if (bridge->asset && bridge->asset_loader) {
        bridge->asset_loader->destroyAsset(bridge->asset);
    }
    bridge->asset = nullptr;
    bridge->glb_bytes.clear();
}

void set_error(FilamentBridge* bridge, const char* message) {
    if (bridge) {
        bridge->last_error = message;
    }
}

}  // namespace

FilamentBridge* filament_bridge_create(void* shared_gl_context) {
    auto bridge = std::make_unique<FilamentBridge>();
    bridge->engine = filament::Engine::Builder()
            .backend(filament::Engine::Backend::OPENGL)
            .sharedContext(shared_gl_context)
            .build();
    if (!bridge->engine) {
        bridge->last_error = "Filament Engine creation failed; OpenXR GL context must be current";
        return bridge.release();
    }
    bridge->materials = filament::gltfio::createJitShaderProvider(bridge->engine);
    bridge->texture_provider = filament::gltfio::createStbProvider(bridge->engine);
    if (!bridge->materials || !bridge->texture_provider) {
        set_error(bridge.get(), "Filament gltfio material or texture provider creation failed");
        return bridge.release();
    }
    filament::gltfio::AssetConfiguration config{bridge->engine, bridge->materials};
    bridge->asset_loader = filament::gltfio::AssetLoader::create(config);
    if (!bridge->asset_loader) {
        set_error(bridge.get(), "Filament AssetLoader creation failed");
    }
    return bridge.release();
}

void filament_bridge_destroy(FilamentBridge* bridge) {
    if (!bridge) return;
    destroy_asset(bridge);
    if (bridge->asset_loader) filament::gltfio::AssetLoader::destroy(&bridge->asset_loader);
    if (bridge->materials) {
        bridge->materials->destroyMaterials();
        delete bridge->materials;
    }
    delete bridge->texture_provider;
    if (bridge->engine) filament::Engine::destroy(&bridge->engine);
    delete bridge;
}

int filament_bridge_load_glb(FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count) {
    if (!bridge || !bridge->engine || !bridge->asset_loader || !bytes || !byte_count) return 0;
    destroy_asset(bridge);
    bridge->last_error.clear();
    bridge->glb_bytes.assign(bytes, bytes + byte_count);
    bridge->asset = bridge->asset_loader->createAsset(bridge->glb_bytes.data(), byte_count);
    if (!bridge->asset) {
        set_error(bridge, "Filament could not parse GLB");
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{bridge->engine, nullptr, true};
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", bridge->texture_provider);
    resources.addTextureProvider("image/jpeg", bridge->texture_provider);
    if (!resources.loadResources(bridge->asset)) {
        destroy_asset(bridge);
        set_error(bridge, "Filament could not load GLB resources");
        return 0;
    }
    bridge->asset->releaseSourceData();
    return 1;
}

int filament_bridge_apply_animations(FilamentBridge* bridge, double time_seconds) {
    if (!bridge || !bridge->asset) return 0;
    auto* animator = bridge->asset->getInstance()->getAnimator();
    if (!animator) return 1;
    const size_t count = animator->getAnimationCount();
    for (size_t index = 0; index < count; ++index) {
        const float duration = animator->getAnimationDuration(index);
        const float time = duration > 0.0f
                ? std::fmod(static_cast<float>(time_seconds), duration)
                : 0.0f;
        animator->applyAnimation(index, std::max(0.0f, time));
    }
    animator->updateBoneMatrices();
    return 1;
}

uint32_t filament_bridge_animation_count(const FilamentBridge* bridge) {
    if (!bridge || !bridge->asset) return 0;
    auto* animator = bridge->asset->getInstance()->getAnimator();
    return animator ? static_cast<uint32_t>(animator->getAnimationCount()) : 0;
}

float filament_bridge_animation_duration(const FilamentBridge* bridge, uint32_t animation_index) {
    if (!bridge || !bridge->asset) return 0.0f;
    auto* animator = bridge->asset->getInstance()->getAnimator();
    if (!animator || animation_index >= animator->getAnimationCount()) return 0.0f;
    return animator->getAnimationDuration(animation_index);
}

const char* filament_bridge_last_error(const FilamentBridge* bridge) {
    return bridge ? bridge->last_error.c_str() : "bridge is null";
}
