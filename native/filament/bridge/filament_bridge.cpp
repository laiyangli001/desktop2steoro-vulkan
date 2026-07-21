#include "filament_bridge.h"

#include <algorithm>
#include <array>
#include <cctype>
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
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
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

namespace {

using VulkanPlatform = filament::backend::VulkanPlatform;
using VkImage = ::VkImage;
using VkSemaphore = ::VkSemaphore;

constexpr uint32_t kInvalidImageIndex = UINT32_MAX;

struct PreviewScreenVertex {
    filament::math::float3 position;
    filament::math::float2 uv;
};

class OpenXrVulkanPlatform final : public VulkanPlatform {
public:
    struct ExternalSwapChain final : Platform::SwapChain {
        std::vector<VkImage> images;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{0, 0};
        uint32_t pending_image = kInvalidImageIndex;
        VkSemaphore pending_ready_semaphore = VK_NULL_HANDLE;
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
        out_sync->imageReadySemaphore = swapchain->pending_ready_semaphore;
        swapchain->pending_ready_semaphore = VK_NULL_HANDLE;
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

struct FilamentEyeTarget {
    filament::Renderer* renderer = nullptr;
    filament::View* view = nullptr;
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
    utils::Entity screen_entity;
    filament::VertexBuffer* screen_vertex_buffer = nullptr;
    filament::IndexBuffer* screen_index_buffer = nullptr;
    filament::Material* screen_material = nullptr;
    filament::MaterialInstance* screen_material_instance = nullptr;
    bool screen_in_scene = false;
    filament::Texture* screen_texture = nullptr;
    std::array<filament::Texture*, 2> screen_textures{};
    std::array<std::vector<ScreenTextureSlot>, 2> screen_texture_cache;
    filament::TextureSampler screen_texture_sampler;
    std::vector<PreviewScreenVertex> screen_vertices;
    std::vector<uint16_t> screen_indices;
    filament::backend::VulkanPlatform::VulkanSharedContext shared_context{};
    MaterialBrightnessState brightness;
    std::array<ControllerAsset, 2> controllers;
    std::array<FilamentEyeTarget, 2> eyes;
    uint32_t active_eye = 0;
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
    uint32_t diagnostic_frame_count = 0;
    bool frame_active = false;
};

void activate_eye(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->eyes.size()) return;
    auto& eye = bridge->eyes[eye_index];
    bridge->active_eye = eye_index;
    bridge->renderer = eye.renderer;
    bridge->view = eye.view;
    bridge->camera = eye.camera;
    bridge->color_grading = eye.color_grading;
    bridge->swapchain = eye.swapchain;
    bridge->external_swapchain = eye.external_swapchain;
    bridge->frame_active = eye.frame_active;
    bridge->screen_texture = bridge->screen_textures[eye_index];
    if (bridge->screen_texture && bridge->screen_material_instance) {
        bridge->screen_material_instance->setParameter(
                "screenTexture", bridge->screen_texture,
                bridge->screen_texture_sampler);
    }
}

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
    utils::Entity screen_entity;
    filament::VertexBuffer* screen_vertex_buffer = nullptr;
    filament::IndexBuffer* screen_index_buffer = nullptr;
    filament::Material* screen_material = nullptr;
    filament::MaterialInstance* screen_material_instance = nullptr;
    std::vector<PreviewScreenVertex> screen_vertices;
    std::vector<uint16_t> screen_indices;
    MaterialBrightnessState brightness{ {}, {}, 2.0f, 1.0f };
    std::vector<uint8_t> glb_bytes;
    std::string last_error;
};

namespace {

template<typename Target>
bool configure_color_pipeline(Target* target) {
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

void set_error(FilamentBridge* bridge, const char* message) {
    if (bridge) {
        bridge->last_error = message;
    }
}

void destroy_controller_asset(FilamentBridge* bridge, ControllerAsset& controller) {
    if (controller.asset && bridge->scene) {
        bridge->scene->removeEntities(
                controller.asset->getEntities(), controller.asset->getEntityCount());
    }
    if (controller.asset && bridge->asset_loader) {
        bridge->asset_loader->destroyAsset(controller.asset);
    }
    controller = {};
}

std::string controller_semantic(std::string name) {
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
    return {};
}

filament::math::mat4f interpolate_controller_transform(
        const filament::math::mat4f& minimum,
        const filament::math::mat4f& maximum,
        float amount) {
    const float t = std::clamp(std::abs(amount), 0.0f, 1.0f);
    const auto& target = amount < 0.0f ? minimum : maximum;
    filament::math::mat4f result = target;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result[column][row] = minimum[column][row] +
                    (target[column][row] - minimum[column][row]) * t;
        }
    }
    return result;
}

float controller_animation_amount(
        const ControllerAsset& controller, const std::string& semantic) {
    if (semantic == "trigger") return controller.trigger;
    if (semantic == "grip") return controller.grip;
    if (semantic == "joystick_x") return controller.joystick_x;
    if (semantic == "joystick_y") return controller.joystick_y;
    if (semantic == "joystick") return controller.joystick_x != 0.0f ||
            controller.joystick_y != 0.0f ? 1.0f : 0.0f;
    if (semantic == "a_button") return (controller.button_mask & (1u << 0)) ? 1.0f : 0.0f;
    if (semantic == "b_button") return (controller.button_mask & (1u << 1)) ? 1.0f : 0.0f;
    if (semantic == "x_button") return (controller.button_mask & (1u << 2)) ? 1.0f : 0.0f;
    if (semantic == "y_button") return (controller.button_mask & (1u << 3)) ? 1.0f : 0.0f;
    if (semantic == "menu_button") return (controller.button_mask & (1u << 4)) ? 1.0f : 0.0f;
    return 0.0f;
}

void update_controller_animations(
        FilamentBridge* bridge, ControllerAsset& controller) {
    if (!controller.asset || !bridge->engine) return;
    auto& transforms = bridge->engine->getTransformManager();
    for (const auto& animation : controller.animations) {
        const float amount = controller_animation_amount(controller, animation.semantic);
        if (!transforms.hasComponent(animation.value_entity)) continue;
        transforms.setTransform(
                transforms.getInstance(animation.value_entity),
                interpolate_controller_transform(
                        animation.min_transform, animation.max_transform, amount));
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

void destroy_preview_screen(FilamentPreview* preview) {
    if (!preview || !preview->engine) return;
    if (preview->scene && !preview->screen_entity.isNull()) {
        preview->scene->remove(preview->screen_entity);
    }
    if (!preview->screen_entity.isNull()) {
        preview->engine->destroy(preview->screen_entity);
        preview->screen_entity = {};
    }
    if (preview->screen_vertex_buffer) {
        preview->engine->destroy(preview->screen_vertex_buffer);
        preview->screen_vertex_buffer = nullptr;
    }
    if (preview->screen_index_buffer) {
        preview->engine->destroy(preview->screen_index_buffer);
        preview->screen_index_buffer = nullptr;
    }
    if (preview->screen_material_instance) {
        preview->engine->destroy(preview->screen_material_instance);
        preview->screen_material_instance = nullptr;
    }
    if (preview->screen_material) {
        preview->engine->destroy(preview->screen_material);
        preview->screen_material = nullptr;
    }
    preview->screen_vertices.clear();
    preview->screen_indices.clear();
}

int update_preview_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees) {
    if (!preview || !preview->engine || !preview->screen_vertex_buffer ||
            !std::isfinite(position_x) || !std::isfinite(position_y) ||
            !std::isfinite(position_z) || !std::isfinite(width) ||
            !std::isfinite(height) || width <= 0.0f || height <= 0.0f ||
            !std::isfinite(rotation_x_degrees) || !std::isfinite(rotation_y_degrees) ||
            !std::isfinite(rotation_z_degrees)) return 0;
    constexpr float kPi = 3.14159265358979323846f;
    const float yaw = rotation_x_degrees * kPi / 180.0f;
    const float pitch = rotation_y_degrees * kPi / 180.0f;
    const float roll = rotation_z_degrees * kPi / 180.0f;
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cr = std::cos(roll), sr = std::sin(roll);
    const filament::math::float3 right{
            cy * cr + sy * sp * sr, sr * cp, -sy * cr + cy * sp * sr};
    const filament::math::float3 up{
            -cy * sr + sy * sp * cr, cr * cp, sr * sy + cy * sp * cr};
    const filament::math::float3 center{position_x, position_y, position_z};
    const filament::math::float3 half_right = right * (width * 0.5f);
    const filament::math::float3 half_up = up * (height * 0.5f);
    preview->screen_vertices = {
            {center - half_right - half_up, {0.0f, 0.0f}},
            {center + half_right - half_up, {1.0f, 0.0f}},
            {center - half_right + half_up, {0.0f, 1.0f}},
            {center + half_right + half_up, {1.0f, 1.0f}},
    };
    preview->screen_vertex_buffer->setBufferAt(*preview->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    preview->screen_vertices.data(),
                    preview->screen_vertices.size() * sizeof(PreviewScreenVertex), nullptr));
    return 1;
}

int create_preview_screen(FilamentPreview* preview) {
    if (!preview || !preview->engine || !preview->scene) return 0;
    const char* shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            float2 uv = getUV0();
            float2 grid_uv = abs(fract(uv * float2(16.0, 9.0)) - 0.5);
            float line = step(0.47, max(grid_uv.x, grid_uv.y));
            float3 base = float3(0.1, 0.45, 1.0);
            float3 grid = mix(base, float3(0.72, 0.88, 1.0), line * 0.35);
            material.baseColor = float4(grid, 0.72);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S Preview Screen")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::TRANSPARENT)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false)
            // The preview screen is a virtual display layer; do not let the
            // environment depth buffer hide it.
            .depthCulling(false)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    const filamat::Package package = builder.build(preview->engine->getJobSystem());
    if (!package.isValid()) {
        set_preview_error(preview, "Filament could not build preview screen material");
        return 0;
    }
    preview->screen_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*preview->engine);
    if (!preview->screen_material) {
        set_preview_error(preview, "Filament could not create preview screen material");
        return 0;
    }
    preview->screen_material_instance = preview->screen_material->createInstance();
    if (!preview->screen_material_instance) {
        set_preview_error(preview, "Filament could not create preview screen material instance");
        return 0;
    }
    preview->screen_vertices.resize(4);
    preview->screen_indices = {0, 1, 2, 1, 3, 2};
    preview->screen_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(4)
            .bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(PreviewScreenVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(PreviewScreenVertex))
            .build(*preview->engine);
    preview->screen_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(static_cast<uint32_t>(preview->screen_indices.size()))
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*preview->engine);
    if (!preview->screen_vertex_buffer || !preview->screen_index_buffer) {
        set_preview_error(preview, "Filament could not create preview screen geometry");
        return 0;
    }
    preview->screen_index_buffer->setBuffer(*preview->engine,
            filament::IndexBuffer::BufferDescriptor(
                    preview->screen_indices.data(),
                    preview->screen_indices.size() * sizeof(uint16_t), nullptr));
    preview->screen_entity = utils::EntityManager::get().create();
    const auto result = filament::RenderableManager::Builder(1)
            .boundingBox({{-20000.0f, -20000.0f, -20000.0f}, {20000.0f, 20000.0f, 20000.0f}})
            .material(0, preview->screen_material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    preview->screen_vertex_buffer, preview->screen_index_buffer,
                    0, static_cast<uint32_t>(preview->screen_indices.size()))
            .priority(7)
            .culling(false)
            .castShadows(false)
            .receiveShadows(false)
            .build(*preview->engine, preview->screen_entity);
    if (result != filament::RenderableManager::Builder::Success) {
        set_preview_error(preview, "Filament could not create preview screen renderable");
        return 0;
    }
    preview->scene->addEntity(preview->screen_entity);
    return 1;
}

void destroy_bridge_screen(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    if (bridge->scene && bridge->screen_in_scene && !bridge->screen_entity.isNull()) {
        bridge->scene->remove(bridge->screen_entity);
    }
    bridge->screen_in_scene = false;
    if (!bridge->screen_entity.isNull()) {
        bridge->engine->destroy(bridge->screen_entity);
        bridge->screen_entity = {};
    }
    if (bridge->screen_vertex_buffer) {
        bridge->engine->destroy(bridge->screen_vertex_buffer);
        bridge->screen_vertex_buffer = nullptr;
    }
    if (bridge->screen_index_buffer) {
        bridge->engine->destroy(bridge->screen_index_buffer);
        bridge->screen_index_buffer = nullptr;
    }
    if (bridge->screen_material_instance) {
        bridge->engine->destroy(bridge->screen_material_instance);
        bridge->screen_material_instance = nullptr;
    }
    for (auto& cache : bridge->screen_texture_cache) {
        for (auto& slot : cache) {
            if (slot.texture) {
                bridge->engine->destroy(slot.texture);
                slot.texture = nullptr;
            }
        }
        cache.clear();
    }
    bridge->screen_textures = {};
    bridge->screen_texture = nullptr;
    if (bridge->screen_material) {
        bridge->engine->destroy(bridge->screen_material);
        bridge->screen_material = nullptr;
    }
    bridge->screen_vertices.clear();
    bridge->screen_indices.clear();
}

int update_bridge_screen(
        FilamentBridge* bridge,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees) {
    if (!bridge || !bridge->engine || !bridge->screen_vertex_buffer ||
            !std::isfinite(position_x) || !std::isfinite(position_y) ||
            !std::isfinite(position_z) || !std::isfinite(width) ||
            !std::isfinite(height) || width <= 0.0f || height <= 0.0f ||
            !std::isfinite(rotation_x_degrees) || !std::isfinite(rotation_y_degrees) ||
            !std::isfinite(rotation_z_degrees)) return 0;
    constexpr float kPi = 3.14159265358979323846f;
    const float yaw = rotation_x_degrees * kPi / 180.0f;
    const float pitch = rotation_y_degrees * kPi / 180.0f;
    const float roll = rotation_z_degrees * kPi / 180.0f;
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float cr = std::cos(roll), sr = std::sin(roll);
    const filament::math::float3 right{
            cy * cr + sy * sp * sr, sr * cp, -sy * cr + cy * sp * sr};
    const filament::math::float3 up{
            -cy * sr + sy * sp * cr, cr * cp, sr * sy + cy * sp * cr};
    const filament::math::float3 center{position_x, position_y, position_z};
    const filament::math::float3 half_right = right * (width * 0.5f);
    const filament::math::float3 half_up = up * (height * 0.5f);
    bridge->screen_vertices = {
            {center - half_right - half_up, {0.0f, 0.0f}},
            {center + half_right - half_up, {1.0f, 0.0f}},
            {center - half_right + half_up, {0.0f, 1.0f}},
            {center + half_right + half_up, {1.0f, 1.0f}},
    };
    bridge->screen_vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    bridge->screen_vertices.data(),
                    bridge->screen_vertices.size() * sizeof(PreviewScreenVertex), nullptr));
    return 1;
}

int create_bridge_screen(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine || !bridge->scene) return 0;
    destroy_bridge_screen(bridge);
    const char* shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            material.baseColor = texture(materialParams_screenTexture, getUV0());
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S OpenXR Screen")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .parameter("screenTexture", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::TRANSPARENT)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false)
            .depthCulling(false)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    const filamat::Package package = builder.build(bridge->engine->getJobSystem());
    if (!package.isValid()) {
        set_error(bridge, "Filament could not build OpenXR screen material");
        return 0;
    }
    bridge->screen_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*bridge->engine);
    if (!bridge->screen_material) {
        set_error(bridge, "Filament could not create OpenXR screen material");
        return 0;
    }
    bridge->screen_material_instance = bridge->screen_material->createInstance();
    bridge->screen_vertices.resize(4);
    bridge->screen_indices = {0, 1, 2, 1, 3, 2};
    bridge->screen_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(4).bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(PreviewScreenVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(PreviewScreenVertex))
            .build(*bridge->engine);
    bridge->screen_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(static_cast<uint32_t>(bridge->screen_indices.size()))
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*bridge->engine);
    if (!bridge->screen_material_instance || !bridge->screen_vertex_buffer ||
            !bridge->screen_index_buffer) {
        set_error(bridge, "Filament could not create OpenXR screen geometry");
        return 0;
    }
    bridge->screen_index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    bridge->screen_indices.data(),
                    bridge->screen_indices.size() * sizeof(uint16_t), nullptr));
    bridge->screen_entity = utils::EntityManager::get().create();
    const auto result = filament::RenderableManager::Builder(1)
            .boundingBox({{-20000.0f, -20000.0f, -20000.0f}, {20000.0f, 20000.0f, 20000.0f}})
            .material(0, bridge->screen_material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    bridge->screen_vertex_buffer, bridge->screen_index_buffer,
                    0, static_cast<uint32_t>(bridge->screen_indices.size()))
            .priority(7).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, bridge->screen_entity);
    if (result != filament::RenderableManager::Builder::Success) {
        set_error(bridge, "Filament could not create OpenXR screen renderable");
        return 0;
    }
    // The sampler is required by the material. Keep the renderable detached
    // until a valid runtime Vulkan image has been imported.
    return 1;
}

int set_bridge_screen_image(FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->screen_material_instance ||
            !image || width == 0 || height == 0) return 0;
    if (format != VK_FORMAT_R8G8B8A8_UNORM &&
            format != VK_FORMAT_R8G8B8A8_SRGB) {
        set_error(bridge, "Unsupported virtual screen Vulkan image format");
        return 0;
    }
    const uint32_t eye_index = bridge->active_eye;
    for (const auto& slot : bridge->screen_texture_cache[eye_index]) {
        if (slot.image == image && slot.width == width &&
                slot.height == height && slot.format == format && slot.texture) {
            bridge->screen_textures[eye_index] = slot.texture;
            bridge->screen_texture = slot.texture;
            bridge->screen_material_instance->setParameter(
                    "screenTexture", slot.texture, bridge->screen_texture_sampler);
            if (!bridge->screen_in_scene && !bridge->screen_entity.isNull()) {
                bridge->scene->addEntity(bridge->screen_entity);
                bridge->screen_in_scene = true;
            }
            return 1;
        }
    }
    auto* texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            // Runtime eye images contain display-referred sRGB bytes in a
            // Vulkan SRGB image; decode them exactly once on sample.
            .format(filament::Texture::InternalFormat::SRGB8_A8)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .usage(filament::Texture::Usage::SAMPLEABLE)
            .import(reinterpret_cast<intptr_t>(const_cast<void*>(image)))
            .build(*bridge->engine);
    if (!texture) {
        set_error(bridge, "Filament could not import virtual screen Vulkan image");
        return 0;
    }
    bridge->screen_texture_cache[eye_index].push_back(
            ScreenTextureSlot{image, texture, width, height, format});
    bridge->screen_textures[eye_index] = texture;
    bridge->screen_texture = texture;
    bridge->screen_material_instance->setParameter(
            "screenTexture", bridge->screen_texture,
            bridge->screen_texture_sampler);
    if (!bridge->screen_in_scene && !bridge->screen_entity.isNull()) {
        bridge->scene->addEntity(bridge->screen_entity);
        bridge->screen_in_scene = true;
    }
    return 1;
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
    bridge->scene = bridge->engine->createScene();
    bridge->materials = filament::gltfio::createJitShaderProvider(bridge->engine);
    bridge->texture_provider = filament::gltfio::createStbProvider(bridge->engine);
    if (!bridge->scene || !bridge->materials ||
            !bridge->texture_provider) {
        set_error(bridge.get(), "Filament Vulkan resource creation failed");
        return bridge.release();
    }
    for (auto& eye : bridge->eyes) {
        eye.renderer = bridge->engine->createRenderer();
        eye.view = bridge->engine->createView();
        eye.camera = bridge->engine->createCamera(
                utils::EntityManager::get().create());
        if (!eye.renderer || !eye.view || !eye.camera) {
            set_error(bridge.get(), "Filament Vulkan eye resource creation failed");
            return bridge.release();
        }
        eye.camera->lookAt(
                filament::math::float3{0.0f, 0.0f, 3.0f},
                filament::math::float3{0.0f, 0.0f, 0.0f},
                filament::math::float3{0.0f, 1.0f, 0.0f});
        eye.view->setScene(bridge->scene);
        eye.view->setCamera(eye.camera);
    }
    activate_eye(bridge.get(), 0);
    for (auto& eye : bridge->eyes) {
        bridge->view = eye.view;
        bridge->camera = eye.camera;
        bridge->color_grading = nullptr;
        if (!configure_color_pipeline(bridge.get())) {
            set_error(bridge.get(), "Filament Vulkan color pipeline creation failed");
            return bridge.release();
        }
        eye.color_grading = bridge->color_grading;
    }
    activate_eye(bridge.get(), 0);
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
    for (auto& controller : bridge->controllers) {
        destroy_controller_asset(bridge, controller);
    }
    destroy_bridge_screen(bridge);
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

int filament_bridge_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height) {
    return filament_bridge_create_eye_swapchain(
            bridge, 0, image_handles, image_count, format, width, height);
}

int filament_bridge_create_eye_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height) {
    if (!bridge || !bridge->engine || !bridge->platform ||
            eye_index >= bridge->eyes.size()) return 0;
    auto& eye = bridge->eyes[eye_index];
    if (eye.swapchain) {
        bridge->engine->destroy(eye.swapchain);
        eye.swapchain = nullptr;
        eye.external_swapchain = nullptr;
    }
    auto* external = bridge->platform->create_external_swapchain(
            image_handles, image_count, static_cast<VkFormat>(format), width, height);
    if (!external) {
        set_error(bridge, "Invalid OpenXR Vulkan swapchain image list");
        return 0;
    }
    uint64_t swapchain_flags = 0;
    if (static_cast<VkFormat>(format) == VK_FORMAT_R8G8B8A8_SRGB ||
            static_cast<VkFormat>(format) == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchain_flags = filament::SwapChain::CONFIG_SRGB_COLORSPACE;
    }
    eye.swapchain = bridge->engine->createSwapChain(external, swapchain_flags);
    if (!eye.swapchain) {
        bridge->platform->destroy(external);
        set_error(bridge, "Filament Vulkan SwapChain creation failed");
        return 0;
    }
    eye.external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    std::fprintf(stderr,
            "[FilamentBridge] eye swapchain created eye=%u images=%u format=%d "
            "extent=%ux%u first_image=%p\n",
            eye_index, image_count, format, width, height,
            image_count ? reinterpret_cast<void*>(
                    reinterpret_cast<uintptr_t>(eye.external_swapchain->images[0]))
                         : nullptr);
    std::fflush(stderr);
    eye.camera->setProjection(
            45.0,
            static_cast<double>(width) / static_cast<double>(height),
            0.05,
            1000.0);
    eye.view->setViewport(filament::Viewport{0, 0, width, height});
    activate_eye(bridge, eye_index);
    return 1;
}

int filament_bridge_set_active_eye(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->eyes.size()) return 0;
    if (bridge->frame_active || bridge->eyes[eye_index].frame_active) return 0;
    activate_eye(bridge, eye_index);
    return 1;
}

int filament_bridge_set_acquired_image(FilamentBridge* bridge, uint32_t image_index) {
    if (!bridge || !bridge->swapchain || !bridge->platform) return 0;
    const int result = bridge->platform->set_pending_image(
            bridge->external_swapchain, image_index) ? 1 : 0;
    if (bridge->diagnostic_frame_count < 8) {
        const auto& images = bridge->external_swapchain->images;
        const void* image = image_index < images.size()
                ? reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(images[image_index]))
                : nullptr;
        std::fprintf(stderr,
                "[FilamentBridge] acquired eye=%u index=%u image=%p result=%d\n",
                bridge->active_eye, image_index, image, result);
        std::fflush(stderr);
    }
    return result;
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
    bridge->eyes[bridge->active_eye].frame_active = bridge->frame_active;
    if (!bridge->frame_active) {
        set_error(bridge, "Filament Renderer::beginFrame failed");
    }
    if (bridge->diagnostic_frame_count < 8) {
        std::fprintf(stderr,
                "[FilamentBridge] begin eye=%u renderer=%p swapchain=%p active=%d\n",
                bridge->active_eye, static_cast<void*>(bridge->renderer),
                static_cast<void*>(bridge->swapchain), bridge->frame_active ? 1 : 0);
        std::fflush(stderr);
    }
    bridge->renderer->render(bridge->view);
    return bridge->frame_active ? 1 : 0;
}

int filament_bridge_end_frame(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->frame_active) return 0;
    bridge->renderer->endFrame();
    bridge->frame_active = false;
    bridge->eyes[bridge->active_eye].frame_active = false;
    if (!bridge->engine) return 0;
    // Queue this eye and let the presenter synchronize once after both eyes.
    // Waiting here serialized the two eye submissions and caused avoidable
    // frame stalls in the projection-layer path.
    bridge->engine->flush();
    if (bridge->diagnostic_frame_count < 8) {
        std::fprintf(stderr, "[FilamentBridge] end eye=%u\n", bridge->active_eye);
        std::fflush(stderr);
        if (bridge->active_eye == 1) {
            ++bridge->diagnostic_frame_count;
        }
    }
    return 1;
}

int filament_bridge_wait_for_idle(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return 0;
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
    collect_material_brightness(bridge, true);
    bridge->asset->releaseSourceData();
    bridge->engine->flushAndWait();
    bridge->glb_bytes.clear();
    return 1;
}

int filament_bridge_load_controller(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count) {
    if (!bridge || !bridge->engine || !bridge->asset_loader ||
            hand > 1 || !bytes || !byte_count) {
        return 0;
    }
    auto& controller = bridge->controllers[hand];
    destroy_controller_asset(bridge, controller);
    controller.bytes.assign(bytes, bytes + byte_count);
    controller.asset = bridge->asset_loader->createAsset(
            controller.bytes.data(), byte_count);
    if (!controller.asset) {
        set_error(bridge, "Filament could not parse controller GLB");
        controller = {};
        return 0;
    }
    filament::gltfio::ResourceConfiguration config{bridge->engine, nullptr, true};
    filament::gltfio::ResourceLoader resources(config);
    resources.addTextureProvider("image/png", bridge->texture_provider);
    resources.addTextureProvider("image/jpeg", bridge->texture_provider);
    if (!resources.loadResources(controller.asset)) {
        destroy_controller_asset(bridge, controller);
        set_error(bridge, "Filament could not load controller GLB resources");
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
        const std::string semantic = controller_semantic(value_name);
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
    update_controller_animations(bridge, controller);
    return 1;
}

int filament_bridge_set_controller_pose(
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

int filament_bridge_set_controller_inputs(
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
    update_controller_animations(bridge, controller);
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
    // Keep the desktop preview target aligned with the validated OpenXR sRGB
    // swapchain path and the shared Rec709 color-grading output.
    preview->swapchain = preview->engine->createSwapChain(
            native_window, filament::SwapChain::CONFIG_SRGB_COLORSPACE);
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
    if (!configure_color_pipeline(preview.get())) {
        set_preview_error(preview.get(), "Filament preview color pipeline creation failed");
        return preview.release();
    }
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
    destroy_preview_screen(preview);
    if (preview->scene && !preview->fill_light.isNull()) {
        preview->scene->remove(preview->fill_light);
    }
    if (!preview->fill_light.isNull() && preview->engine) {
        preview->engine->destroy(preview->fill_light);
    }
    if (preview->swapchain && preview->engine) preview->engine->destroy(preview->swapchain);
    if (preview->color_grading && preview->engine) {
        preview->view->setColorGrading(nullptr);
        preview->engine->destroy(preview->color_grading);
    }
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
    if (!preview->screen_material_instance && !create_preview_screen(preview)) {
        destroy_preview_asset(preview);
        return 0;
    }
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

int filament_preview_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees) {
    return update_preview_screen(preview, position_x, position_y, position_z,
            width, height, rotation_x_degrees, rotation_y_degrees, rotation_z_degrees);
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

int filament_bridge_set_fill_light(
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

int filament_bridge_create_screen(FilamentBridge* bridge) {
    return create_bridge_screen(bridge);
}

int filament_bridge_set_screen(
        FilamentBridge* bridge,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees) {
    return update_bridge_screen(bridge, position_x, position_y, position_z,
            width, height, rotation_x_degrees, rotation_y_degrees, rotation_z_degrees);
}

int filament_bridge_set_screen_image(FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format) {
    return set_bridge_screen_image(bridge, image, width, height, format);
}

int filament_bridge_set_screen_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore) {
    if (!bridge || !bridge->platform || !bridge->swapchain || !semaphore) return 0;
    const auto ready = reinterpret_cast<VkSemaphore>(
            const_cast<void*>(semaphore));
    return bridge->platform->set_pending_ready_semaphore(bridge->swapchain, ready)
            ? 1 : 0;
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
