#include "filament_bridge.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <iterator>
#include <vector>

#include <backend/platforms/VulkanPlatform.h>
#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <filamat/MaterialBuilder.h>
#include <gltfio/Animator.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/TextureProvider.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <utils/EntityManager.h>

namespace {

using VulkanPlatform = filament::backend::VulkanPlatform;
using VkImage = ::VkImage;

constexpr uint32_t kInvalidImageIndex = UINT32_MAX;

class OpenXrVulkanPlatform final : public VulkanPlatform {
public:
    struct ExternalSwapChain final : Platform::SwapChain {
        std::vector<VkImage> images;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{0, 0};
        uint32_t pending_image = kInvalidImageIndex;
        uint32_t current_image = kInvalidImageIndex;
    };

    SwapChainPtr create_external_swapchain(
            const void* const* image_handles,
            uint32_t image_count,
            VkFormat format,
            uint32_t width,
            uint32_t height) {
        if (!image_handles || image_count == 0 || width == 0 || height == 0) {
            return nullptr;
        }
        auto swapchain = std::make_unique<ExternalSwapChain>();
        swapchain->images.reserve(image_count);
        for (uint32_t index = 0; index < image_count; ++index) {
            const auto image = reinterpret_cast<VkImage>(
                    const_cast<void*>(image_handles[index]));
            if (image == VK_NULL_HANDLE) {
                return nullptr;
            }
            swapchain->images.push_back(image);
        }
        swapchain->format = format;
        swapchain->extent = {width, height};
        return swapchain.release();
    }

    bool set_pending_image(SwapChainPtr handle, uint32_t image_index) noexcept {
        auto* swapchain = as_external(handle);
        if (!swapchain || image_index >= swapchain->images.size()) {
            return false;
        }
        swapchain->pending_image = image_index;
        return true;
    }

    bool set_pending_image(uint32_t image_index) noexcept {
        return set_pending_image(m_active_swapchain, image_index);
    }

    SwapChainPtr createSwapChain(void* native_window, uint64_t,
            VkExtent2D) override {
        auto* swapchain = static_cast<ExternalSwapChain*>(native_window);
        m_active_swapchain = swapchain;
        return swapchain;
    }

    SwapChainBundle getSwapChainBundle(SwapChainPtr handle) override {
        SwapChainBundle bundle;
        auto* swapchain = as_external(handle);
        if (!swapchain) {
            return bundle;
        }
        bundle.colors.reserve(swapchain->images.size());
        for (VkImage image : swapchain->images) {
            bundle.colors.push_back(image);
        }
        bundle.colorFormat = swapchain->format;
        bundle.extent = swapchain->extent;
        bundle.layerCount = 1;
        return bundle;
    }

    VkResult acquire(SwapChainPtr handle, ImageSyncData* out_sync) override {
        auto* swapchain = as_external(handle);
        if (!swapchain || !out_sync ||
                swapchain->pending_image == kInvalidImageIndex) {
            return VK_NOT_READY;
        }
        swapchain->current_image = swapchain->pending_image;
        swapchain->pending_image = kInvalidImageIndex;
        out_sync->imageIndex = swapchain->current_image;
        out_sync->imageReadySemaphore = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }

    VkResult present(SwapChainPtr handle, uint32_t image_index,
            VkSemaphore finished_drawing) override {
        (void) finished_drawing;
        auto* swapchain = as_external(handle);
        if (!swapchain || image_index >= swapchain->images.size()) {
            return VK_ERROR_OUT_OF_DATE_KHR;
        }
        return VK_SUCCESS;
    }

    bool hasResized(SwapChainPtr) override { return false; }

    bool isProtected(SwapChainPtr) override { return false; }

    Customization getCustomization() const noexcept override {
        Customization customization;
        customization.transitionSwapChainImageLayoutForPresent = false;
        return customization;
    }

    VkResult recreate(SwapChainPtr) override { return VK_SUCCESS; }

    void destroy(SwapChainPtr handle) override {
        if (as_external(handle) == m_active_swapchain) {
            m_active_swapchain = nullptr;
        }
        delete as_external(handle);
    }

    void terminate() override { VulkanPlatform::terminate(); }

protected:
    ExtensionSet getSwapchainInstanceExtensions() const override { return {}; }

    SurfaceBundle createVkSurfaceKHR(void*, VkInstance,
            uint64_t) const noexcept override {
        return {VK_NULL_HANDLE, {0, 0}};
    }

private:
    ExternalSwapChain* m_active_swapchain = nullptr;

    static ExternalSwapChain* as_external(SwapChainPtr handle) noexcept {
        return static_cast<ExternalSwapChain*>(handle);
    }
};

}  // namespace

struct MaterialBrightnessEntry {
    filament::MaterialInstance* material = nullptr;
    filament::math::float4 base_color_factor{};
};

struct MaterialBrightnessState {
    std::vector<MaterialBrightnessEntry> scene_materials;
    std::vector<MaterialBrightnessEntry> skybox_materials;
    float scene_exposure_ev = 0.0f;
    float skybox_brightness = 1.0f;
};

struct StarGlimVertex {
    float position[3];
    float uv[2];
};

struct FilamentBridge {
    filament::Engine* engine = nullptr;
    filament::Renderer* renderer = nullptr;
    filament::Scene* scene = nullptr;
    filament::View* view = nullptr;
    filament::Camera* camera = nullptr;
    filament::gltfio::MaterialProvider* materials = nullptr;
    filament::gltfio::TextureProvider* texture_provider = nullptr;
    filament::gltfio::AssetLoader* asset_loader = nullptr;
    filament::gltfio::FilamentAsset* asset = nullptr;
    filament::SwapChain* swapchain = nullptr;
    OpenXrVulkanPlatform::ExternalSwapChain* external_swapchain = nullptr;
    OpenXrVulkanPlatform* platform = nullptr;
    filament::backend::VulkanPlatform::VulkanSharedContext shared_context{};
    MaterialBrightnessState brightness;
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
    bool frame_active = false;
};

struct FilamentPreview {
    filament::Engine* engine = nullptr;
    filament::Renderer* renderer = nullptr;
    filament::Scene* scene = nullptr;
    filament::View* view = nullptr;
    filament::Camera* camera = nullptr;
    filament::gltfio::MaterialProvider* materials = nullptr;
    filament::gltfio::TextureProvider* texture_provider = nullptr;
    filament::gltfio::AssetLoader* asset_loader = nullptr;
    filament::gltfio::FilamentAsset* asset = nullptr;
    filament::SwapChain* swapchain = nullptr;
    utils::Entity fill_light;
    utils::Entity star_glim_entity;
    filament::VertexBuffer* star_glim_vertex_buffer = nullptr;
    filament::IndexBuffer* star_glim_index_buffer = nullptr;
    filament::Material* star_glim_material = nullptr;
    filament::MaterialInstance* star_glim_material_instance = nullptr;
    filament::Texture* star_glim_stars = nullptr;
    filament::Texture* star_glim_mask = nullptr;
    std::vector<StarGlimVertex> star_glim_vertices;
    std::vector<uint16_t> star_glim_indices;
    MaterialBrightnessState brightness{ {}, {}, 2.0f, 1.0f };
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
};

namespace {

void set_error(FilamentBridge* bridge, const char* message) {
    if (bridge) {
        bridge->last_error = message;
    }
}

void destroy_asset(FilamentBridge* bridge) {
    if (bridge->asset && bridge->scene) {
        bridge->scene->removeEntities(
                bridge->asset->getEntities(), bridge->asset->getEntityCount());
    }
    if (bridge->asset && bridge->asset_loader) {
        bridge->asset_loader->destroyAsset(bridge->asset);
    }
    bridge->asset = nullptr;
    bridge->brightness.scene_materials.clear();
    bridge->brightness.skybox_materials.clear();
    bridge->glb_bytes.clear();
}

void set_preview_error(FilamentPreview* preview, const char* message) {
    if (preview) {
        preview->last_error = message;
    }
}

void destroy_preview_asset(FilamentPreview* preview) {
    if (preview->asset && preview->scene) {
        preview->scene->removeEntities(
                preview->asset->getEntities(), preview->asset->getEntityCount());
    }
    if (preview->asset && preview->asset_loader) {
        preview->asset_loader->destroyAsset(preview->asset);
    }
    preview->asset = nullptr;
    preview->brightness.scene_materials.clear();
    preview->brightness.skybox_materials.clear();
    preview->glb_bytes.clear();
}

void destroy_star_glim(FilamentPreview* preview) {
    if (!preview || !preview->engine) return;
    if (preview->scene && !preview->star_glim_entity.isNull()) {
        preview->scene->remove(preview->star_glim_entity);
    }
    if (!preview->star_glim_entity.isNull()) {
        preview->engine->destroy(preview->star_glim_entity);
        preview->star_glim_entity = {};
    }
    if (preview->star_glim_material_instance) {
        preview->engine->destroy(preview->star_glim_material_instance);
        preview->star_glim_material_instance = nullptr;
    }
    if (preview->star_glim_material) {
        preview->engine->destroy(preview->star_glim_material);
        preview->star_glim_material = nullptr;
    }
    if (preview->star_glim_vertex_buffer) {
        preview->engine->destroy(preview->star_glim_vertex_buffer);
        preview->star_glim_vertex_buffer = nullptr;
    }
    if (preview->star_glim_index_buffer) {
        preview->engine->destroy(preview->star_glim_index_buffer);
        preview->star_glim_index_buffer = nullptr;
    }
    if (preview->star_glim_stars) {
        preview->engine->destroy(preview->star_glim_stars);
        preview->star_glim_stars = nullptr;
    }
    if (preview->star_glim_mask) {
        preview->engine->destroy(preview->star_glim_mask);
        preview->star_glim_mask = nullptr;
    }
    preview->star_glim_vertices.clear();
    preview->star_glim_indices.clear();
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
void apply_material_brightness(BridgeType* bridge) {
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
void collect_material_brightness(BridgeType* bridge, bool enable_fill_channel) {
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
    apply_material_brightness(bridge);
}

}  // namespace

FilamentBridge* filament_bridge_create_vulkan(
        const FilamentBridgeVulkanCreateInfo* info) {
    auto bridge = std::make_unique<FilamentBridge>();
    if (!info || !info->instance || !info->physical_device || !info->device) {
        set_error(bridge.get(), "Vulkan create info contains a null handle");
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
        set_error(bridge.get(), "Filament Vulkan Engine creation failed");
        delete bridge->platform;
        bridge->platform = nullptr;
        return bridge.release();
    }
    bridge->renderer = bridge->engine->createRenderer();
    bridge->scene = bridge->engine->createScene();
    bridge->view = bridge->engine->createView();
    bridge->camera = bridge->engine->createCamera(
            utils::EntityManager::get().create());
    bridge->materials = filament::gltfio::createJitShaderProvider(bridge->engine);
    bridge->texture_provider = filament::gltfio::createStbProvider(bridge->engine);
    if (!bridge->renderer || !bridge->scene || !bridge->view || !bridge->camera ||
            !bridge->materials || !bridge->texture_provider) {
        set_error(bridge.get(), "Filament Vulkan resource creation failed");
        return bridge.release();
    }
    bridge->camera->lookAt(
            filament::math::float3{0.0f, 0.0f, 3.0f},
            filament::math::float3{0.0f, 0.0f, 0.0f},
            filament::math::float3{0.0f, 1.0f, 0.0f});
    bridge->view->setScene(bridge->scene);
    bridge->view->setCamera(bridge->camera);
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
    if (bridge->swapchain && bridge->engine) {
        bridge->engine->destroy(bridge->swapchain);
    }
    if (bridge->view && bridge->engine) {
        bridge->engine->destroy(bridge->view);
    }
    if (bridge->camera && bridge->engine) {
        bridge->engine->destroy(bridge->camera->getEntity());
    }
    if (bridge->scene && bridge->engine) {
        bridge->engine->destroy(bridge->scene);
    }
    if (bridge->renderer && bridge->engine) {
        bridge->engine->destroy(bridge->renderer);
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

int filament_bridge_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height) {
    if (!bridge || !bridge->engine || !bridge->platform) return 0;
    if (bridge->swapchain) {
        bridge->engine->destroy(bridge->swapchain);
        bridge->swapchain = nullptr;
        bridge->external_swapchain = nullptr;
    }
    auto* external = bridge->platform->create_external_swapchain(
            image_handles, image_count, static_cast<VkFormat>(format), width, height);
    if (!external) {
        set_error(bridge, "Invalid OpenXR Vulkan swapchain image list");
        return 0;
    }
    bridge->swapchain = bridge->engine->createSwapChain(external);
    if (!bridge->swapchain) {
        bridge->platform->destroy(external);
        set_error(bridge, "Filament Vulkan SwapChain creation failed");
        return 0;
    }
    bridge->external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    bridge->camera->setProjection(
            45.0,
            static_cast<double>(width) / static_cast<double>(height),
            0.05,
            1000.0);
    bridge->view->setViewport(filament::Viewport{0, 0, width, height});
    return 1;
}

int filament_bridge_set_acquired_image(FilamentBridge* bridge, uint32_t image_index) {
    if (!bridge || !bridge->swapchain || !bridge->platform) return 0;
    return bridge->platform->set_pending_image(
            bridge->external_swapchain, image_index) ? 1 : 0;
}

int filament_bridge_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    if (!bridge || !bridge->camera) return 0;
    bridge->camera->lookAt(
            filament::math::float3{eye_x, eye_y, eye_z},
            filament::math::float3{center_x, center_y, center_z},
            filament::math::float3{up_x, up_y, up_z});
    return 1;
}

int filament_bridge_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    if (!bridge || !bridge->camera || vertical_fov_degrees <= 0.0 ||
            aspect <= 0.0 || near_plane <= 0.0 || far_plane <= near_plane) {
        return 0;
    }
    bridge->camera->setProjection(
            vertical_fov_degrees, aspect, near_plane, far_plane);
    return 1;
}

int filament_bridge_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane) {
    if (!bridge || !bridge->camera || right <= left || top <= bottom ||
            near_plane <= 0.0 || far_plane <= near_plane) {
        return 0;
    }
    bridge->camera->setProjection(
            filament::Camera::Projection::PERSPECTIVE,
            left, right, bottom, top, near_plane, far_plane);
    return 1;
}

int filament_bridge_begin_frame(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->swapchain || bridge->frame_active) {
        return 0;
    }
    bridge->frame_active = bridge->renderer->beginFrame(bridge->swapchain);
    if (!bridge->frame_active) {
        set_error(bridge, "Filament Renderer::beginFrame failed");
    }
    bridge->renderer->render(bridge->view);
    return bridge->frame_active ? 1 : 0;
}

int filament_bridge_end_frame(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->frame_active) return 0;
    bridge->renderer->endFrame();
    bridge->frame_active = false;
    if (!bridge->engine) return 0;
    bridge->engine->flushAndWait();
    return 1;
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
    bridge->scene->addEntities(
            bridge->asset->getEntities(), bridge->asset->getEntityCount());
    collect_material_brightness(bridge, false);
    bridge->asset->releaseSourceData();
    bridge->engine->flushAndWait();
    bridge->glb_bytes.clear();
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

FilamentPreview* filament_preview_create(void* native_window, uint32_t width, uint32_t height) {
    auto preview = std::make_unique<FilamentPreview>();
    if (!native_window || width == 0 || height == 0) {
        set_preview_error(preview.get(), "Preview window or dimensions are invalid");
        return preview.release();
    }
    preview->engine = filament::Engine::Builder()
            .backend(filament::Engine::Backend::DEFAULT)
            .build();
    if (!preview->engine) {
        set_preview_error(preview.get(), "Filament preview Engine creation failed");
        return preview.release();
    }
    preview->renderer = preview->engine->createRenderer();
    preview->scene = preview->engine->createScene();
    preview->view = preview->engine->createView();
    preview->camera = preview->engine->createCamera(
            utils::EntityManager::get().create());
    preview->materials = filament::gltfio::createJitShaderProvider(preview->engine);
    preview->texture_provider = filament::gltfio::createStbProvider(preview->engine);
    preview->swapchain = preview->engine->createSwapChain(native_window);
    if (!preview->renderer || !preview->scene || !preview->view || !preview->camera ||
            !preview->materials || !preview->texture_provider || !preview->swapchain) {
        set_preview_error(preview.get(), "Filament preview resource creation failed");
        return preview.release();
    }
    preview->camera->lookAt(
            filament::math::float3{0.0f, 0.0f, 3.0f},
            filament::math::float3{0.0f, 0.0f, 0.0f},
            filament::math::float3{0.0f, 1.0f, 0.0f});
    preview->view->setScene(preview->scene);
    preview->view->setCamera(preview->camera);
    preview->view->setViewport(filament::Viewport{0, 0, width, height});
    preview->fill_light = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
            .color(filament::LinearColor{1.0f, 0.88f, 0.78f})
            .intensity(100000.0f)
            .direction({-0.35f, -1.0f, -0.55f})
            .lightChannel(0, false)
            .lightChannel(1, true)
            .castShadows(false)
            .build(*preview->engine, preview->fill_light);
    preview->scene->addEntity(preview->fill_light);
    filament::gltfio::AssetConfiguration config{preview->engine, preview->materials};
    preview->asset_loader = filament::gltfio::AssetLoader::create(config);
    if (!preview->asset_loader) {
        set_preview_error(preview.get(), "Filament preview AssetLoader creation failed");
    }
    return preview.release();
}

void filament_preview_destroy(FilamentPreview* preview) {
    if (!preview) return;
    destroy_preview_asset(preview);
    destroy_star_glim(preview);
    if (preview->scene && !preview->fill_light.isNull()) {
        preview->scene->remove(preview->fill_light);
    }
    if (!preview->fill_light.isNull() && preview->engine) {
        preview->engine->destroy(preview->fill_light);
    }
    if (preview->swapchain && preview->engine) preview->engine->destroy(preview->swapchain);
    if (preview->view && preview->engine) preview->engine->destroy(preview->view);
    if (preview->camera && preview->engine) preview->engine->destroy(preview->camera->getEntity());
    if (preview->scene && preview->engine) preview->engine->destroy(preview->scene);
    if (preview->renderer && preview->engine) preview->engine->destroy(preview->renderer);
    if (preview->asset_loader) filament::gltfio::AssetLoader::destroy(&preview->asset_loader);
    if (preview->materials) {
        preview->materials->destroyMaterials();
        delete preview->materials;
    }
    delete preview->texture_provider;
    if (preview->engine) filament::Engine::destroy(&preview->engine);
    delete preview;
}

int filament_preview_load_glb(FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count) {
    if (!preview || !preview->engine || !preview->asset_loader || !bytes || !byte_count) return 0;
    destroy_preview_asset(preview);
    preview->last_error.clear();
    preview->glb_bytes.assign(bytes, bytes + byte_count);
    preview->asset = preview->asset_loader->createAsset(preview->glb_bytes.data(), byte_count);
    if (!preview->asset) {
        set_preview_error(preview, "Filament preview could not parse GLB");
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{preview->engine, nullptr, true};
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", preview->texture_provider);
    resources.addTextureProvider("image/jpeg", preview->texture_provider);
    if (!resources.loadResources(preview->asset)) {
        destroy_preview_asset(preview);
        set_preview_error(preview, "Filament preview could not load GLB resources");
        return 0;
    }
    preview->scene->addEntities(
            preview->asset->getEntities(), preview->asset->getEntityCount());
    collect_material_brightness(preview, true);
    preview->asset->releaseSourceData();
    preview->engine->flushAndWait();
    preview->glb_bytes.clear();
    return 1;
}

int filament_preview_apply_animations(FilamentPreview* preview, double time_seconds) {
    if (!preview || !preview->asset || !std::isfinite(time_seconds)) return 0;
    auto* animator = preview->asset->getInstance()->getAnimator();
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

int filament_preview_set_star_glim(
        FilamentPreview* preview,
        const uint8_t* stars_bytes, uint32_t stars_byte_count,
        const uint8_t* mask_bytes, uint32_t mask_byte_count,
        float intensity, float speed, float shine_speed,
        float cell_density, float cell_offset, float cell_soft,
        float cell_value, float strength) {
    if (!preview || !preview->engine || !preview->scene ||
            !preview->texture_provider || !stars_bytes || !stars_byte_count ||
            !mask_bytes || !mask_byte_count) {
        return 0;
    }
    const float values[] = {intensity, speed, shine_speed, cell_density,
            cell_offset, cell_soft, cell_value, strength};
    for (float value : values) {
        if (!std::isfinite(value)) return 0;
    }

    destroy_star_glim(preview);
    preview->star_glim_stars = preview->texture_provider->pushTexture(
            stars_bytes, stars_byte_count, "image/png",
            filament::gltfio::TextureProvider::TextureFlags::sRGB);
    preview->star_glim_mask = preview->texture_provider->pushTexture(
            mask_bytes, mask_byte_count, "image/png",
            filament::gltfio::TextureProvider::TextureFlags::NONE);
    if (!preview->star_glim_stars || !preview->star_glim_mask) {
        destroy_star_glim(preview);
        set_preview_error(preview, "Filament could not decode StarGlim textures");
        return 0;
    }
    preview->texture_provider->waitForCompletion();
    preview->texture_provider->updateQueue();
    while (preview->texture_provider->popTexture()) {}

    const char* shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            const float3 skyCenter = float3(-24.519, 56.915, -962.746);
            float3 direction = normalize(getWorldPosition() - skyCenter);
            float2 panoUv = float2(
                    atan(direction.z, direction.x) * 0.1591549431 + 0.5,
                    asin(clamp(direction.y, -1.0, 1.0)) * 0.3183098862 + 0.5);
            float2 sampleUv = panoUv;
            float3 textureStars = texture(materialParams_stars, sampleUv).rgb;
            float mask = texture(materialParams_mask, panoUv).r;
            float2 cellUv = panoUv * materialParams.cellDensity + materialParams.cellOffset;
            float2 cell = floor(cellUv);
            float2 local = fract(cellUv) - 0.5;
            float phase = fract(sin(dot(cell, float2(12.9898, 78.233))) * 43758.5453);
            float localSpeed = mix(0.45, 1.8, fract(phase * 7.13));
            float pulse = 0.5 + 0.5 * sin(
                    materialParams.time * materialParams.shineSpeed * localSpeed
                    + phase * 6.2831853);
            float threshold = clamp(materialParams.cellValue + (1.0 - materialParams.cellSoft), 0.0, 1.0);
            float starRadius = mix(0.055, 0.16, fract(phase * 19.17));
            float diamond = 1.0 - smoothstep(0.0, starRadius, abs(local.x) + abs(local.y));
            float horizontalRay = (1.0 - smoothstep(0.0, 0.018, abs(local.y)))
                    * (1.0 - smoothstep(0.0, starRadius * 2.8, abs(local.x)));
            float verticalRay = (1.0 - smoothstep(0.0, 0.018, abs(local.x)))
                    * (1.0 - smoothstep(0.0, starRadius * 2.8, abs(local.y)));
            float starShape = max(diamond, max(horizontalRay, verticalRay) * 0.9);
            float proceduralStar = starShape * step(0.92, phase);
            float textureStar = max(max(textureStars.r, textureStars.g), textureStars.b) * 4.0;
            float twinkle = smoothstep(threshold, 1.0, pulse) * materialParams.strength;
            float skyOnly = smoothstep(0.0, 0.08, panoUv.y);
            float3 textureBackgroundColor = textureStars * 8.0 * mask * materialParams.intensity;
            float textureBackgroundAlpha = textureStar * mask * skyOnly;
            float3 twinkleColor = float3(1.0, 0.88, 0.68)
                    * proceduralStar * twinkle * mask * materialParams.intensity;
            float twinkleAlpha = proceduralStar * twinkle * mask * skyOnly;
            material.baseColor = float4(
                    textureBackgroundColor + twinkleColor,
                    textureBackgroundAlpha + twinkleAlpha);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S StarGlim")
            .material(shader)
            .parameter("stars", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("mask", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("time", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("intensity", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("speed", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("shineSpeed", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("cellDensity", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("cellOffset", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("cellSoft", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("cellValue", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("strength", filamat::MaterialBuilder::UniformType::FLOAT)
            .require(filament::VertexAttribute::UV0)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::ADD)
            .doubleSided(true)
            .depthWrite(false)
            .depthCulling(true)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    filamat::Package package = builder.build(preview->engine->getJobSystem());
    if (!package.isValid()) {
        destroy_star_glim(preview);
        set_preview_error(preview, "Filament could not build StarGlim material");
        return 0;
    }
    preview->star_glim_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*preview->engine);
    if (!preview->star_glim_material) {
        destroy_star_glim(preview);
        set_preview_error(preview, "Filament could not create StarGlim material");
        return 0;
    }
    preview->star_glim_material_instance = preview->star_glim_material->createInstance();
    if (!preview->star_glim_material_instance) {
        destroy_star_glim(preview);
        set_preview_error(preview, "Filament could not create StarGlim material instance");
        return 0;
    }

    // Fixed geometry matching the Artemis SkyBox bounds. It remains in the room.
    const float min_x = -924.519f;
    const float max_x = 875.481f;
    const float min_y = -843.085f;
    const float max_y = 956.915f;
    const float min_z = -1862.746f;
    const float max_z = -62.746f;
    const StarGlimVertex vertices[] = {
        {{min_x, min_y, min_z}, {0, 0}}, {{max_x, min_y, min_z}, {1, 0}},
        {{max_x, max_y, min_z}, {1, 1}}, {{min_x, max_y, min_z}, {0, 1}},
        {{max_x, min_y, max_z}, {0, 0}}, {{min_x, min_y, max_z}, {1, 0}},
        {{min_x, max_y, max_z}, {1, 1}}, {{max_x, max_y, max_z}, {0, 1}},
        {{min_x, min_y, max_z}, {0, 0}}, {{min_x, min_y, min_z}, {1, 0}},
        {{min_x, max_y, min_z}, {1, 1}}, {{min_x, max_y, max_z}, {0, 1}},
        {{max_x, min_y, min_z}, {0, 0}}, {{max_x, min_y, max_z}, {1, 0}},
        {{max_x, max_y, max_z}, {1, 1}}, {{max_x, max_y, min_z}, {0, 1}},
        {{min_x, max_y, min_z}, {0, 0}}, {{max_x, max_y, min_z}, {1, 0}},
        {{max_x, max_y, max_z}, {1, 1}}, {{min_x, max_y, max_z}, {0, 1}},
        {{min_x, min_y, max_z}, {0, 0}}, {{max_x, min_y, max_z}, {1, 0}},
        {{max_x, min_y, min_z}, {1, 1}}, {{min_x, min_y, min_z}, {0, 1}},
    };
    const uint16_t indices[] = {
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    };
    preview->star_glim_vertices.assign(std::begin(vertices), std::end(vertices));
    preview->star_glim_indices.assign(std::begin(indices), std::end(indices));
    preview->star_glim_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(static_cast<uint32_t>(preview->star_glim_vertices.size()))
            .bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(StarGlimVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(StarGlimVertex))
            .build(*preview->engine);
    preview->star_glim_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(static_cast<uint32_t>(preview->star_glim_indices.size()))
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*preview->engine);
    if (!preview->star_glim_vertex_buffer || !preview->star_glim_index_buffer) {
        destroy_star_glim(preview);
        set_preview_error(preview, "Filament could not create StarGlim geometry");
        return 0;
    }
    preview->star_glim_vertex_buffer->setBufferAt(*preview->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    preview->star_glim_vertices.data(),
                    preview->star_glim_vertices.size() * sizeof(StarGlimVertex), nullptr));
    preview->star_glim_index_buffer->setBuffer(*preview->engine,
            filament::IndexBuffer::BufferDescriptor(
                    preview->star_glim_indices.data(),
                    preview->star_glim_indices.size() * sizeof(uint16_t), nullptr));
    filament::TextureSampler sampler(
            filament::TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR,
            filament::TextureSampler::MagFilter::LINEAR);
    preview->star_glim_material_instance->setParameter(
            "stars", preview->star_glim_stars, sampler);
    preview->star_glim_material_instance->setParameter(
            "mask", preview->star_glim_mask, sampler);
    preview->star_glim_material_instance->setParameter("time", 0.0f);
    preview->star_glim_material_instance->setParameter("intensity", intensity);
    preview->star_glim_material_instance->setParameter("speed", speed);
    preview->star_glim_material_instance->setParameter("shineSpeed", shine_speed);
    preview->star_glim_material_instance->setParameter("cellDensity", cell_density);
    preview->star_glim_material_instance->setParameter("cellOffset", cell_offset);
    preview->star_glim_material_instance->setParameter("cellSoft", cell_soft);
    preview->star_glim_material_instance->setParameter("cellValue", cell_value);
    preview->star_glim_material_instance->setParameter("strength", strength);
    preview->star_glim_entity = utils::EntityManager::get().create();
    filament::RenderableManager::Builder(1)
            .boundingBox({{min_x, min_y, min_z}, {max_x, max_y, max_z}})
            .material(0, preview->star_glim_material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    preview->star_glim_vertex_buffer, preview->star_glim_index_buffer,
                    0, static_cast<uint32_t>(preview->star_glim_indices.size()))
            .culling(false)
            .castShadows(false)
            .receiveShadows(false)
            .build(*preview->engine, preview->star_glim_entity);
    preview->scene->addEntity(preview->star_glim_entity);
    return 1;
}

int filament_preview_set_star_glim_time(FilamentPreview* preview, double time_seconds) {
    if (!preview || !preview->star_glim_material_instance || !std::isfinite(time_seconds)) return 0;
    preview->star_glim_material_instance->setParameter("time", static_cast<float>(time_seconds));
    return 1;
}

int filament_preview_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    if (!preview || !preview->camera) return 0;
    preview->camera->lookAt(
            filament::math::float3{eye_x, eye_y, eye_z},
            filament::math::float3{center_x, center_y, center_z},
            filament::math::float3{up_x, up_y, up_z});
    return 1;
}

int filament_preview_set_projection(
        FilamentPreview* preview,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    if (!preview || !preview->camera || vertical_fov_degrees <= 0.0 ||
            aspect <= 0.0 || near_plane <= 0.0 || far_plane <= near_plane) return 0;
    preview->camera->setProjection(
            vertical_fov_degrees, aspect, near_plane, far_plane);
    return 1;
}

int filament_preview_set_viewport(FilamentPreview* preview, uint32_t width, uint32_t height) {
    if (!preview || !preview->view || width == 0 || height == 0) return 0;
    preview->view->setViewport(filament::Viewport{0, 0, width, height});
    return 1;
}

int filament_preview_set_scene_exposure(FilamentPreview* preview, float exposure_ev) {
    if (!preview || !preview->engine || !std::isfinite(exposure_ev)) {
        return 0;
    }
    preview->brightness.scene_exposure_ev = std::clamp(exposure_ev, -8.0f, 8.0f);
    apply_material_brightness(preview);
    return 1;
}

int filament_preview_set_fill_light(
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

int filament_preview_set_skybox_brightness(FilamentPreview* preview, float brightness) {
    if (!preview || !std::isfinite(brightness) || brightness < 0.0f) return 0;
    preview->brightness.skybox_brightness = std::min(brightness, 16.0f);
    apply_material_brightness(preview);
    return 1;
}

int filament_bridge_set_scene_exposure(FilamentBridge* bridge, float exposure_ev) {
    if (!bridge || !std::isfinite(exposure_ev)) return 0;
    bridge->brightness.scene_exposure_ev = std::clamp(exposure_ev, -8.0f, 8.0f);
    apply_material_brightness(bridge);
    return 1;
}

int filament_bridge_set_skybox_brightness(FilamentBridge* bridge, float brightness) {
    if (!bridge || !std::isfinite(brightness) || brightness < 0.0f) return 0;
    bridge->brightness.skybox_brightness = std::min(brightness, 16.0f);
    apply_material_brightness(bridge);
    return 1;
}

int filament_preview_render(FilamentPreview* preview) {
    if (!preview || !preview->renderer || !preview->swapchain || !preview->view) return 0;
    if (!preview->renderer->beginFrame(preview->swapchain)) return 1;
    preview->renderer->render(preview->view);
    preview->renderer->endFrame();
    return 1;
}

const char* filament_preview_last_error(const FilamentPreview* preview) {
    return preview ? preview->last_error.c_str() : "preview is null";
}
