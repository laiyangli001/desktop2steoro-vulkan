#pragma once

#include "filament_bridge.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <iterator>
#include <vector>

#include <backend/platforms/VulkanPlatform.h>
#include <filament/Camera.h>
#include <filament/ColorGrading.h>
#include <filament/ColorSpace.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/RenderTarget.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/SwapChain.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>
#include <filament/TransformManager.h>
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
#include <math/vec2.h>
#include <utils/EntityManager.h>

using VulkanPlatform = filament::backend::VulkanPlatform;
using VkImage = ::VkImage;
using VkSemaphore = ::VkSemaphore;

constexpr uint32_t kInvalidImageIndex = UINT32_MAX;
constexpr uint8_t kScreenLayerBase = 3;
constexpr uint8_t kScreenMipCopyLayerBase = 5;

struct PreviewScreenVertex {
    filament::math::float3 position;
    filament::math::float2 uv;
};

struct GlowVertex {
    filament::math::float3 position;
    filament::math::float2 uv;
    filament::math::float2 effect;
};

struct MsdfTextVertex {
    filament::math::float3 position;
    filament::math::float2 uv;
    filament::math::float4 color;
};

struct MsdfTextPage {
    utils::Entity entity;
    filament::Material* material = nullptr;
    filament::MaterialInstance* material_instance = nullptr;
    filament::Texture* texture = nullptr;
    filament::TextureSampler sampler;
    filament::VertexBuffer* vertex_buffer = nullptr;
    filament::IndexBuffer* index_buffer = nullptr;
    std::vector<MsdfTextVertex> vertices;
    std::vector<uint16_t> indices;
};

class OpenXrVulkanPlatform final : public VulkanPlatform {
public:
    struct ExternalSwapChain final : Platform::SwapChain {
        std::vector<VkImage> images;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{0, 0};
        uint32_t layer_count = 1;
        uint32_t pending_image = kInvalidImageIndex;
        VkSemaphore pending_ready_semaphore = VK_NULL_HANDLE;
        uint32_t current_image = kInvalidImageIndex;
        VkSemaphore last_finished_drawing = VK_NULL_HANDLE;
    };

    SwapChainPtr create_external_swapchain(
            const void* const* image_handles,
            uint32_t image_count,
            VkFormat format,
            uint32_t width,
            uint32_t height,
            uint32_t layer_count = 1) {
        if (!image_handles || image_count == 0 || width == 0 || height == 0 ||
                layer_count == 0) {
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
        swapchain->layer_count = layer_count;
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

    bool set_pending_ready_semaphore(SwapChainPtr handle, VkSemaphore semaphore) noexcept {
        auto* swapchain = as_external(handle);
        if (!swapchain) return false;
        swapchain->pending_ready_semaphore = semaphore;
        return true;
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
        bundle.layerCount = swapchain->layer_count;
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
        out_sync->imageReadySemaphore = swapchain->pending_ready_semaphore;
        swapchain->pending_ready_semaphore = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }

    VkResult present(SwapChainPtr handle, uint32_t image_index,
            VkSemaphore finished_drawing) override {
        auto* swapchain = as_external(handle);
        if (!swapchain || image_index >= swapchain->images.size()) {
            return VK_ERROR_OUT_OF_DATE_KHR;
        }
        swapchain->last_finished_drawing = finished_drawing;
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

struct MaterialBrightnessEntry {
    filament::MaterialInstance* material = nullptr;
    filament::math::float4 base_color_factor{};
};

struct MaterialBrightnessState {
    std::vector<MaterialBrightnessEntry> scene_materials;
    std::vector<MaterialBrightnessEntry> skybox_materials;
    std::vector<utils::Entity> skybox_entities;
    float scene_exposure_ev = 0.0f;
    float skybox_brightness = 1.0f;
};

struct FilamentEyeTarget {
    filament::Renderer* renderer = nullptr;
    filament::View* view = nullptr;
    // Effect, controller, and guide views provide an explicit screen -> Glow
    // -> controller -> guide ordering after the room view.
    filament::View* foreground_view = nullptr;
    filament::View* controller_view = nullptr;
    filament::View* controller_guide_view = nullptr;
    filament::Camera* camera = nullptr;
    filament::ColorGrading* color_grading = nullptr;
    filament::SwapChain* swapchain = nullptr;
    OpenXrVulkanPlatform::ExternalSwapChain* external_swapchain = nullptr;
    bool frame_active = false;
};

struct ControllerAnimation {
    utils::Entity value_entity;
    utils::Entity min_entity;
    utils::Entity max_entity;
    filament::math::mat4f value_transform;
    filament::math::mat4f min_transform;
    filament::math::mat4f max_transform;
    std::string semantic;
    std::string value_name;
};

struct ControllerAsset {
    filament::gltfio::FilamentAsset* asset = nullptr;
    std::vector<uint8_t> bytes;
    std::vector<ControllerAnimation> animations;
    float trigger = 0.0f;
    float grip = 0.0f;
    float joystick_x = 0.0f;
    float joystick_y = 0.0f;
    uint32_t button_mask = 0;
    std::array<float, 7> button_values{};
    std::chrono::steady_clock::time_point last_input_time{};
    bool input_initialized = false;
    bool visible = true;
};

struct ScreenTextureSlot {
    const void* image = nullptr;
    filament::Texture* texture = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t format = VK_FORMAT_UNDEFINED;
};

struct FilamentBridge {
    filament::Engine* engine = nullptr;
    filament::Renderer* renderer = nullptr;
    filament::Scene* scene = nullptr;
    filament::View* view = nullptr;
    filament::Camera* camera = nullptr;
    filament::ColorGrading* color_grading = nullptr;
    filament::gltfio::MaterialProvider* materials = nullptr;
    filament::gltfio::TextureProvider* texture_provider = nullptr;
    filament::gltfio::AssetLoader* asset_loader = nullptr;
    filament::gltfio::FilamentAsset* asset = nullptr;
    filament::SwapChain* swapchain = nullptr;
    OpenXrVulkanPlatform::ExternalSwapChain* external_swapchain = nullptr;
    OpenXrVulkanPlatform* platform = nullptr;
    utils::Entity fill_light;
    utils::Entity controller_top_light;
    filament::IndirectLight* indirect_light = nullptr;
    filament::IndirectLight* controller_indirect_light = nullptr;
    filament::Scene* foreground_scene = nullptr;
    utils::Entity screen_light;
    filament::math::float3 screen_light_position{0.0f, 0.0f, 0.0f};
    filament::math::float3 screen_light_direction{0.0f, 0.0f, -1.0f};
    float screen_light_falloff = 2.0f;
    std::array<utils::Entity, 2> screen_entities{};
    utils::Entity screen_entity;
    filament::VertexBuffer* screen_vertex_buffer = nullptr;
    filament::IndexBuffer* screen_index_buffer = nullptr;
    filament::Material* screen_material = nullptr;
    std::array<filament::MaterialInstance*, 2> screen_material_instances{};
    filament::MaterialInstance* screen_material_instance = nullptr;
    bool screen_in_scene = false;
    bool passthrough_backdrop = false;
    filament::Texture* screen_texture = nullptr;
    std::array<filament::Texture*, 2> screen_source_textures{};
    std::array<int32_t, 2> screen_source_formats{
            VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB};
    std::array<filament::Texture*, 2> screen_textures{};
    std::array<std::vector<ScreenTextureSlot>, 2> screen_texture_cache;
    filament::TextureSampler screen_texture_sampler;
    filament::TextureSampler screen_source_texture_sampler;
    std::array<filament::Texture*, 2> screen_lanczos_textures{};
    std::array<filament::RenderTarget*, 2> screen_lanczos_render_targets{};
    std::array<filament::Texture*, 2> screen_mip_textures{};
    std::array<filament::RenderTarget*, 2> screen_mip_render_targets{};
    std::array<bool, 2> screen_mip_ready{};
    std::array<uint64_t, 2> screen_source_bind_count{};
    std::array<uint64_t, 2> screen_mip_generation_count{};
    std::array<uint64_t, 2> screen_source_versions{};
    std::array<uint64_t, 2> screen_last_mip_versions{};
    filament::Texture* screen_fixed_source_texture = nullptr;
    std::array<filament::MaterialInstance*, 2>
            screen_mip_copy_material_instances{};
    filament::MaterialInstance* screen_mip_copy_material_instance = nullptr;
    std::array<utils::Entity, 2> screen_mip_copy_entities{};
    utils::Entity screen_mip_copy_entity;
    filament::VertexBuffer* screen_mip_copy_vertex_buffer = nullptr;
    filament::IndexBuffer* screen_mip_copy_index_buffer = nullptr;
    std::array<filament::View*, 2> screen_mip_copy_views{};
    filament::View* screen_mip_copy_view = nullptr;
    filament::Camera* screen_mip_copy_camera = nullptr;
    std::array<PreviewScreenVertex, 4> screen_mip_copy_vertices{};
    std::array<uint16_t, 6> screen_mip_copy_indices{};
    bool screen_mip_experiment_enabled = true;
    // Keep the legacy screen-quality sharpening strength explicit. The
    // external Vulkan image has one mip level, so the screen material uses a
    // footprint-aware filter followed by this bounded RCAS pass.
    float screen_filter_sharpness = 0.35f;
    // Matrix-selected prefilter scale for lower-tier headsets. A value of
    // one preserves the source texel footprint exactly.
    float screen_filter_scale = 1.0f;
    float screen_upscale_scale = 1.0f;
    std::vector<PreviewScreenVertex> screen_vertices;
    std::vector<uint16_t> screen_indices;
    bool screen_curved = false;
    filament::math::float3 screen_center{0.0f, 0.0f, -2.0f};
    filament::math::float3 screen_right{1.0f, 0.0f, 0.0f};
    filament::math::float3 screen_up{0.0f, 1.0f, 0.0f};
    filament::math::float3 screen_forward{0.0f, 0.0f, 1.0f};
    float screen_width = 2.4f;
    float screen_height = 1.35f;
    uint32_t glow_mode = 0;
    filament::math::float3 glow_head_position{0.0f, 0.0f, 0.0f};
    float glow_intensity = 0.175f;
    float glow_width = 0.75f;
    float glow_intensity_multiplier = 0.0f;
    float frosted_intensity = 1.0f;
    float frosted_alpha = 0.42f;
    float frosted_threshold = 0.46f;
    float frosted_lod = 5.4f;
    float frosted_blend = 1.35f;
    float frosted_thickness = 1.6f;
    float frosted_diffuse = 0.85f;
    float frosted_inset = 0.045f;
    float veil_intensity = 1.5f;
    float veil_alpha = 1.0f;
    float glow_shell_intensity_multiplier = 0.0f;
    float glow_shell_radius = 20.0f;
    float glow_shell_height = 9.5f;
    filament::Texture* glow_source_texture = nullptr;
    filament::Texture* glow_cpu_source_texture = nullptr;
    std::vector<ScreenTextureSlot> glow_texture_cache;
    bool glow_source_external = false;
    filament::TextureSampler glow_texture_sampler;
    filament::TextureSampler glow_external_texture_sampler;
    filament::Material* glow_material = nullptr;
    filament::MaterialInstance* glow_outer_material_instance = nullptr;
    filament::MaterialInstance* glow_inner_material_instance = nullptr;
    filament::Material* frost_material = nullptr;
    filament::Material* glow_shell_material = nullptr;
    filament::MaterialInstance* frost_material_instance = nullptr;
    filament::MaterialInstance* veil_material_instance = nullptr;
    filament::MaterialInstance* glow_shell_material_instance = nullptr;
    filament::VertexBuffer* glow_vertex_buffer = nullptr;
    filament::IndexBuffer* glow_index_buffer = nullptr;
    filament::VertexBuffer* frost_vertex_buffer = nullptr;
    filament::IndexBuffer* frost_index_buffer = nullptr;
    filament::VertexBuffer* glow_shell_vertex_buffer = nullptr;
    filament::IndexBuffer* glow_shell_index_buffer = nullptr;
    utils::Entity glow_outer_entity;
    utils::Entity glow_inner_entity;
    utils::Entity frost_entity;
    utils::Entity glow_shell_entity;
    std::vector<GlowVertex> glow_vertices;
    std::vector<uint16_t> glow_indices;
    std::vector<GlowVertex> frost_vertices;
    std::vector<uint16_t> frost_indices;
    std::vector<GlowVertex> glow_shell_vertices;
    std::vector<uint16_t> glow_shell_indices;
    filament::backend::VulkanPlatform::VulkanSharedContext shared_context{};
    MaterialBrightnessState brightness;
    std::array<ControllerAsset, 2> controllers;
    filament::Material* laser_material = nullptr;
    filament::MaterialInstance* laser_material_instance = nullptr;
    filament::VertexBuffer* laser_vertex_buffer = nullptr;
    filament::IndexBuffer* laser_index_buffer = nullptr;
    std::array<utils::Entity, 2> laser_entities{};
    std::array<PreviewScreenVertex, 8> laser_vertices{};
    std::array<uint16_t, 12> laser_indices{};
    utils::Entity controller_guide_entity;
    filament::Material* controller_guide_material = nullptr;
    filament::MaterialInstance* controller_guide_material_instance = nullptr;
    filament::Texture* controller_guide_texture = nullptr;
    filament::TextureSampler controller_guide_texture_sampler;
    filament::VertexBuffer* controller_guide_vertex_buffer = nullptr;
    filament::IndexBuffer* controller_guide_index_buffer = nullptr;
    std::array<PreviewScreenVertex, 4> controller_guide_vertices{};
    std::array<uint16_t, 6> controller_guide_indices{};
    std::array<MsdfTextPage, 4> text_pages{};
    std::array<FilamentEyeTarget, 2> eyes;
    uint32_t active_eye = 0;
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
    uint32_t diagnostic_frame_count = 0;
    bool frame_active = false;
    bool multiview_supported = false;
    bool multiview_active = false;
};

struct FilamentPreview {
    filament::Engine* engine = nullptr;
    filament::Renderer* renderer = nullptr;
    filament::Scene* scene = nullptr;
    filament::View* view = nullptr;
    filament::Camera* camera = nullptr;
    filament::ColorGrading* color_grading = nullptr;
    filament::gltfio::MaterialProvider* materials = nullptr;
    filament::gltfio::TextureProvider* texture_provider = nullptr;
    filament::gltfio::AssetLoader* asset_loader = nullptr;
    filament::gltfio::FilamentAsset* asset = nullptr;
    filament::SwapChain* swapchain = nullptr;
    utils::Entity fill_light;
    filament::IndirectLight* indirect_light = nullptr;
    utils::Entity screen_entity;
    filament::VertexBuffer* screen_vertex_buffer = nullptr;
    filament::IndexBuffer* screen_index_buffer = nullptr;
    filament::Material* screen_material = nullptr;
    filament::MaterialInstance* screen_material_instance = nullptr;
    std::vector<PreviewScreenVertex> screen_vertices;
    std::vector<uint16_t> screen_indices;
    MaterialBrightnessState brightness{ {}, {}, {}, 2.0f, 1.0f };
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
};

void bridge_set_error(FilamentBridge* bridge, const char* message);
void bridge_set_renderable_visible(
        FilamentBridge* bridge, utils::Entity entity, bool visible);
void bridge_set_renderable_layer(
        FilamentBridge* bridge, utils::Entity entity,
        uint8_t layer, bool visible);
