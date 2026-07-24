#include "bridge_screen.h"
#include "bridge_internal.h"

namespace {

constexpr uint32_t kScreenSegments = 48;
constexpr float kCurvedHalfAngle = 0.72f;
constexpr float kLegacyScreenCandelaScale = 1200.0f;

}  // namespace

void bridge_screen_destroy(FilamentBridge* bridge) {
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

int bridge_screen_update(
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
    const filament::math::float3 forward = cross(right, up);
    const filament::math::float3 center{position_x, position_y, position_z};
    bridge->screen_light_position = center;
    bridge->screen_light_direction = -forward;
    bridge->screen_light_falloff =
            std::max(std::sqrt(width * width + height * height), 0.5f);
    if (!bridge->screen_light.isNull()) {
        auto& lights = bridge->engine->getLightManager();
        const auto instance = lights.getInstance(bridge->screen_light);
        if (instance.isValid()) {
            lights.setPosition(instance, center);
            lights.setDirection(instance, bridge->screen_light_direction);
            lights.setFalloff(instance, bridge->screen_light_falloff);
        }
    }
    const float half_width = width * 0.5f;
    const float half_height = height * 0.5f;
    const float radius = half_width / kCurvedHalfAngle;
    bridge->screen_vertices.clear();
    bridge->screen_vertices.reserve((kScreenSegments + 1) * 2);
    for (uint32_t segment = 0; segment <= kScreenSegments; ++segment) {
        const float t = static_cast<float>(segment) /
                static_cast<float>(kScreenSegments);
        float local_x = width * (t - 0.5f);
        float local_z = 0.0f;
        if (bridge->screen_curved) {
            const float angle = -kCurvedHalfAngle + 2.0f * kCurvedHalfAngle * t;
            local_x = radius * std::sin(angle);
            local_z = radius * (1.0f - std::cos(angle));
        }
        const filament::math::float3 column_center =
                center + right * local_x + forward * local_z;
        bridge->screen_vertices.push_back(
                {column_center - up * half_height, {t, 0.0f}});
        bridge->screen_vertices.push_back(
                {column_center + up * half_height, {t, 1.0f}});
    }
    bridge->screen_vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    bridge->screen_vertices.data(),
                    bridge->screen_vertices.size() * sizeof(PreviewScreenVertex), nullptr));
    return 1;
}

int bridge_screen_set_curved(FilamentBridge* bridge, int curved) {
    if (!bridge || !bridge->engine || !bridge->screen_vertex_buffer) return 0;
    bridge->screen_curved = curved != 0;
    return 1;
}

int bridge_screen_set_light(
        FilamentBridge* bridge,
        float red, float green, float blue, float intensity) {
    if (!bridge || !bridge->engine || !bridge->scene ||
            !std::isfinite(red) || !std::isfinite(green) ||
            !std::isfinite(blue) || !std::isfinite(intensity) ||
            red < 0.0f || green < 0.0f || blue < 0.0f || intensity < 0.0f) {
        return 0;
    }
    if (intensity == 0.0f || (red == 0.0f && green == 0.0f && blue == 0.0f)) {
        if (!bridge->screen_light.isNull()) {
            bridge->scene->remove(bridge->screen_light);
            bridge->engine->destroy(bridge->screen_light);
            bridge->screen_light = {};
        }
        return 1;
    }
    if (!bridge->screen_light.isNull()) {
        auto& lights = bridge->engine->getLightManager();
        const auto instance = lights.getInstance(bridge->screen_light);
        if (instance.isValid()) {
            lights.setColor(instance, filament::LinearColor{red, green, blue});
            lights.setIntensityCandela(
                    instance, intensity * kLegacyScreenCandelaScale);
            return 1;
        }
        bridge->scene->remove(bridge->screen_light);
        bridge->engine->destroy(bridge->screen_light);
        bridge->screen_light = {};
    }
    bridge->screen_light = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::FOCUSED_SPOT)
            .color(filament::LinearColor{red, green, blue})
            .intensityCandela(intensity * kLegacyScreenCandelaScale)
            .position(bridge->screen_light_position)
            .direction(bridge->screen_light_direction)
            .falloff(bridge->screen_light_falloff)
            .spotLightCone(1.25f, 1.50f)
            .lightChannel(0, false)
            .lightChannel(1, true)
            .castShadows(false)
            .build(*bridge->engine, bridge->screen_light);
    const auto instance = bridge->engine->getLightManager().getInstance(
            bridge->screen_light);
    if (!instance.isValid()) {
        utils::EntityManager::get().destroy(bridge->screen_light);
        bridge->screen_light = {};
        bridge_set_error(bridge, "Filament could not create virtual screen light");
        return 0;
    }
    bridge->scene->addEntity(bridge->screen_light);
    return 1;
}

int bridge_screen_create(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine || !bridge->scene) return 0;
    bridge_screen_destroy(bridge);
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
        bridge_set_error(bridge, "Filament could not build OpenXR screen material");
        return 0;
    }
    bridge->screen_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*bridge->engine);
    if (!bridge->screen_material) {
        bridge_set_error(bridge, "Filament could not create OpenXR screen material");
        return 0;
    }
    bridge->screen_material_instance = bridge->screen_material->createInstance();
    bridge->screen_vertices.resize((kScreenSegments + 1) * 2);
    bridge->screen_indices.clear();
    bridge->screen_indices.reserve(kScreenSegments * 6);
    for (uint16_t segment = 0; segment < kScreenSegments; ++segment) {
        const uint16_t lower_left = segment * 2;
        const uint16_t upper_left = lower_left + 1;
        const uint16_t lower_right = lower_left + 2;
        const uint16_t upper_right = lower_left + 3;
        bridge->screen_indices.insert(bridge->screen_indices.end(), {
                lower_left, lower_right, upper_left,
                lower_right, upper_right, upper_left});
    }
    bridge->screen_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(static_cast<uint32_t>(bridge->screen_vertices.size()))
            .bufferCount(1)
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
        bridge_set_error(bridge, "Filament could not create OpenXR screen geometry");
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
            // Render the display-referred screen before controllers and laser.
            // The screen does not write depth, so foreground controller PBR
            // geometry and the depth-tested laser remain visible on top.
            .priority(0).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, bridge->screen_entity);
    if (result != filament::RenderableManager::Builder::Success) {
        bridge_set_error(bridge, "Filament could not create OpenXR screen renderable");
        return 0;
    }
    // Display-referred screen content bypasses the HDR scene view.
    bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, false);
    // The sampler is required by the material. Keep the renderable detached
    // until a valid runtime Vulkan image has been imported.
    return 1;
}

int bridge_screen_set_image(FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->screen_material_instance ||
            !image || width == 0 || height == 0) return 0;
    if (format != VK_FORMAT_R8G8B8A8_SRGB) {
        bridge_set_error(bridge,
                "Virtual screen requires VK_FORMAT_R8G8B8A8_SRGB");
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
                bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, true);
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
        bridge_set_error(bridge, "Filament could not import virtual screen Vulkan image");
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
        bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, true);
        bridge->screen_in_scene = true;
    }
    return 1;
}
