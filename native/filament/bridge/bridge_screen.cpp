#include "bridge_screen.h"
#include "bridge_glow.h"
#include "bridge_internal.h"

#include <backend/PixelBufferDescriptor.h>

#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kScreenSegments = 48;
constexpr float kCurvedHalfAngle = 0.72f;
constexpr float kLegacyScreenCandelaScale = 1200.0f;
constexpr float kControllerScreenLightWeight = 0.80f;
// Match the legacy OpenGL runtime-eye sampler bias. Without this, the
// trilinear MIP sampler selects a softer level for the same screen footprint.
constexpr float kLegacyScreenLodBias = -0.35f;

void destroy_screen_mip_target(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->screen_mip_textures.size()) return;
    if (bridge->screen_mip_render_targets[eye_index]) {
        bridge->engine->destroy(bridge->screen_mip_render_targets[eye_index]);
        bridge->screen_mip_render_targets[eye_index] = nullptr;
    }
    if (bridge->screen_mip_textures[eye_index]) {
        bridge->engine->destroy(bridge->screen_mip_textures[eye_index]);
        bridge->screen_mip_textures[eye_index] = nullptr;
    }
    bridge->screen_mip_ready[eye_index] = false;
}

void destroy_screen_lanczos_target(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->screen_lanczos_textures.size()) return;
    if (bridge->screen_lanczos_render_targets[eye_index]) {
        bridge->engine->destroy(bridge->screen_lanczos_render_targets[eye_index]);
        bridge->screen_lanczos_render_targets[eye_index] = nullptr;
    }
    if (bridge->screen_lanczos_textures[eye_index]) {
        bridge->engine->destroy(bridge->screen_lanczos_textures[eye_index]);
        bridge->screen_lanczos_textures[eye_index] = nullptr;
    }
}

void bind_screen_display_texture(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->screen_textures.size()) return;
    const bool use_mip = bridge->screen_mip_experiment_enabled &&
            bridge->screen_mip_ready[eye_index] &&
            bridge->screen_mip_textures[eye_index];
    filament::Texture* texture = use_mip
            ? bridge->screen_mip_textures[eye_index]
            : bridge->screen_source_textures[eye_index];
    bridge->screen_textures[eye_index] = texture;
    if (eye_index == bridge->active_eye) {
        bridge->screen_texture = texture;
    }
    auto* material = bridge->screen_material_instances[eye_index];
    if (texture && material) {
        const auto& sampler = use_mip ? bridge->screen_texture_sampler
                                     : bridge->screen_source_texture_sampler;
        material->setParameter(
                "screenTextureLeft", texture, sampler);
        material->setParameter(
                "screenTextureRight", texture, sampler);
        material->setParameter(
                "screenTexelSize", filament::math::float2{
                        1.0f / static_cast<float>(texture->getWidth()),
                        1.0f / static_cast<float>(texture->getHeight())});
        material->setParameter("screenSourceSize", filament::math::float2{
                static_cast<float>(texture->getWidth()),
                static_cast<float>(texture->getHeight())});
        material->setParameter("screenOutputSize", filament::math::float2{
                static_cast<float>(texture->getWidth()),
                static_cast<float>(texture->getHeight())});
    }
}

void bind_screen_copy_source(
        FilamentBridge* bridge, filament::Texture* source,
        uint32_t width, uint32_t height) {
    if (!bridge || !source || !bridge->screen_mip_copy_material_instance) return;
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTextureLeft", source, bridge->screen_source_texture_sampler);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTextureRight", source, bridge->screen_source_texture_sampler);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexelSize", filament::math::float2{
                    1.0f / static_cast<float>(width),
                    1.0f / static_cast<float>(height)});
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenSourceSize", filament::math::float2{
                    static_cast<float>(width), static_cast<float>(height)});
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenOutputSize", filament::math::float2{
                    static_cast<float>(width), static_cast<float>(height)});
}

bool ensure_screen_mip_target(
        FilamentBridge* bridge, uint32_t eye_index,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->screen_mip_experiment_enabled ||
            eye_index >= bridge->screen_mip_textures.size() ||
            !bridge->screen_mip_copy_views[eye_index] ||
            bridge->screen_mip_copy_entities[eye_index].isNull() ||
            width == 0 || height == 0) {
        return false;
    }
    auto* current = bridge->screen_mip_textures[eye_index];
    if (current && bridge->screen_mip_render_targets[eye_index] &&
            current->getWidth() == width && current->getHeight() == height) {
        return true;
    }
    destroy_screen_mip_target(bridge, eye_index);
    const auto texture_format = format == VK_FORMAT_R8G8B8A8_SRGB
            ? filament::Texture::InternalFormat::SRGB8_A8
            : filament::Texture::InternalFormat::RGBA8;
    using Usage = filament::Texture::Usage;
    const auto usage = static_cast<Usage>(
            static_cast<uint16_t>(Usage::COLOR_ATTACHMENT) |
            static_cast<uint16_t>(Usage::SAMPLEABLE) |
            static_cast<uint16_t>(Usage::BLIT_SRC) |
            static_cast<uint16_t>(Usage::BLIT_DST) |
            static_cast<uint16_t>(Usage::GEN_MIPMAPPABLE));
    uint32_t maximum = std::max(width, height);
    uint8_t levels = 1;
    while (maximum > 1 && levels < 255) {
        maximum = (maximum + 1) / 2;
        ++levels;
    }
    auto* texture = filament::Texture::Builder()
            .width(width).height(height).levels(levels)
            .format(texture_format)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .usage(usage)
            .build(*bridge->engine);
    if (!texture) {
        bridge_set_error(bridge, "Filament could not create screen MIP texture");
        return false;
    }
    auto* target = filament::RenderTarget::Builder()
            .texture(filament::RenderTarget::AttachmentPoint::COLOR, texture)
            .mipLevel(filament::RenderTarget::AttachmentPoint::COLOR, 0)
            .build(*bridge->engine);
    if (!target) {
        bridge->engine->destroy(texture);
        bridge_set_error(bridge, "Filament could not create screen MIP render target");
        return false;
    }
    bridge->screen_mip_textures[eye_index] = texture;
    bridge->screen_mip_render_targets[eye_index] = target;
    // Allocation does not mean LOD 0 contains the current source. The first
    // prepare pass must still render before the display binds this texture.
    bridge->screen_mip_ready[eye_index] = false;
    return true;
}

bool ensure_screen_lanczos_target(
        FilamentBridge* bridge, uint32_t eye_index,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->screen_mip_experiment_enabled ||
            eye_index >= bridge->screen_lanczos_textures.size() ||
            width == 0 || height == 0) {
        return false;
    }
    auto* current = bridge->screen_lanczos_textures[eye_index];
    if (current && bridge->screen_lanczos_render_targets[eye_index] &&
            current->getWidth() == width && current->getHeight() == height) {
        return true;
    }
    destroy_screen_lanczos_target(bridge, eye_index);
    const auto texture_format = format == VK_FORMAT_R8G8B8A8_SRGB
            ? filament::Texture::InternalFormat::SRGB8_A8
            : filament::Texture::InternalFormat::RGBA8;
    using Usage = filament::Texture::Usage;
    const auto usage = static_cast<Usage>(
            static_cast<uint16_t>(Usage::COLOR_ATTACHMENT) |
            static_cast<uint16_t>(Usage::SAMPLEABLE));
    auto* texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            .format(texture_format)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .usage(usage)
            .build(*bridge->engine);
    if (!texture) {
        bridge_set_error(bridge, "Filament could not create screen Lanczos texture");
        return false;
    }
    auto* target = filament::RenderTarget::Builder()
            .texture(filament::RenderTarget::AttachmentPoint::COLOR, texture)
            .build(*bridge->engine);
    if (!target) {
        bridge->engine->destroy(texture);
        bridge_set_error(bridge, "Filament could not create screen Lanczos target");
        return false;
    }
    bridge->screen_lanczos_textures[eye_index] = texture;
    bridge->screen_lanczos_render_targets[eye_index] = target;
    return true;
}

bool ensure_screen_quality_targets(
        FilamentBridge* bridge, uint32_t eye_index,
        uint32_t source_width, uint32_t source_height, int32_t format) {
    if (!bridge || source_width == 0 || source_height == 0) return false;
    const float upscale = std::max(1.0f, bridge->screen_upscale_scale);
    const float downscale = std::max(1.0f, bridge->screen_filter_scale);
    const float scale = upscale > 1.0f ? upscale : 1.0f / downscale;
    const uint32_t width = std::max(
            16u, static_cast<uint32_t>(std::lround(source_width * scale)) & ~1u);
    const uint32_t height = std::max(
            16u, static_cast<uint32_t>(std::lround(source_height * scale)) & ~1u);
    return ensure_screen_lanczos_target(bridge, eye_index, width, height, format) &&
            ensure_screen_mip_target(bridge, eye_index, width, height, format);
}

void add_screen_entities(FilamentBridge* bridge) {
    if (!bridge || !bridge->foreground_scene || bridge->screen_in_scene) return;
    for (uint32_t eye_index = 0; eye_index < bridge->screen_entities.size(); ++eye_index) {
        const auto entity = bridge->screen_entities[eye_index];
        if (entity.isNull()) continue;
        bridge->foreground_scene->addEntity(entity);
        bridge_set_renderable_layer(
                bridge, entity, kScreenLayerBase + eye_index, true);
    }
    bridge->screen_in_scene = true;
}

}  // namespace

void bridge_screen_destroy(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    bridge_glow_destroy(bridge);
    for (auto& entity : bridge->screen_mip_copy_entities) {
        if (bridge->scene && !entity.isNull()) bridge->scene->remove(entity);
        if (!entity.isNull()) bridge->engine->destroy(entity);
        entity = {};
    }
    bridge->screen_mip_copy_entity = {};
    for (auto*& view : bridge->screen_mip_copy_views) {
        if (view) bridge->engine->destroy(view);
        view = nullptr;
    }
    bridge->screen_mip_copy_view = nullptr;
    if (bridge->screen_mip_copy_camera) {
        bridge->engine->destroy(bridge->screen_mip_copy_camera->getEntity());
        bridge->screen_mip_copy_camera = nullptr;
    }
    if (bridge->screen_mip_copy_vertex_buffer) {
        bridge->engine->destroy(bridge->screen_mip_copy_vertex_buffer);
        bridge->screen_mip_copy_vertex_buffer = nullptr;
    }
    if (bridge->screen_mip_copy_index_buffer) {
        bridge->engine->destroy(bridge->screen_mip_copy_index_buffer);
        bridge->screen_mip_copy_index_buffer = nullptr;
    }
    for (auto*& material_instance :
            bridge->screen_mip_copy_material_instances) {
        if (material_instance) {
            bridge->engine->destroy(material_instance);
            material_instance = nullptr;
        }
    }
    bridge->screen_mip_copy_material_instance = nullptr;
    if (bridge->screen_fixed_source_texture) {
        bridge->engine->destroy(bridge->screen_fixed_source_texture);
        bridge->screen_fixed_source_texture = nullptr;
    }
    for (uint32_t eye_index = 0; eye_index < bridge->screen_mip_textures.size(); ++eye_index) {
        destroy_screen_lanczos_target(bridge, eye_index);
        destroy_screen_mip_target(bridge, eye_index);
    }
    for (auto& entity : bridge->screen_entities) {
        if (bridge->foreground_scene && bridge->screen_in_scene && !entity.isNull()) {
            bridge->foreground_scene->remove(entity);
        }
        if (!entity.isNull()) bridge->engine->destroy(entity);
        entity = {};
    }
    bridge->screen_in_scene = false;
    bridge->screen_entity = {};
    if (bridge->screen_vertex_buffer) {
        bridge->engine->destroy(bridge->screen_vertex_buffer);
        bridge->screen_vertex_buffer = nullptr;
    }
    if (bridge->screen_index_buffer) {
        bridge->engine->destroy(bridge->screen_index_buffer);
        bridge->screen_index_buffer = nullptr;
    }
    for (auto*& material_instance : bridge->screen_material_instances) {
        if (material_instance) {
            bridge->engine->destroy(material_instance);
            material_instance = nullptr;
        }
    }
    bridge->screen_material_instance = nullptr;
    for (auto& cache : bridge->screen_texture_cache) {
        for (auto& slot : cache) {
            if (slot.texture) {
                bridge->engine->destroy(slot.texture);
                slot.texture = nullptr;
            }
        }
        cache.clear();
    }
    bridge->screen_source_textures = {};
    bridge->screen_source_formats = {
            VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB};
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
    bridge->screen_center = center;
    bridge->screen_right = right;
    bridge->screen_up = up;
    bridge->screen_forward = forward;
    bridge->screen_width = width;
    bridge->screen_height = height;
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
    bridge_glow_update_geometry(bridge);
    return 1;
}

int bridge_screen_set_curved(FilamentBridge* bridge, int curved) {
    if (!bridge || !bridge->engine || !bridge->screen_vertex_buffer) return 0;
    bridge->screen_curved = curved != 0;
    bridge_glow_update_geometry(bridge);
    return 1;
}

int bridge_screen_set_light(
        FilamentBridge* bridge,
        float red, float green, float blue, float intensity) {
    if (!bridge || !bridge->engine || !bridge->foreground_scene ||
            !std::isfinite(red) || !std::isfinite(green) ||
            !std::isfinite(blue) || !std::isfinite(intensity) ||
            red < 0.0f || green < 0.0f || blue < 0.0f || intensity < 0.0f) {
        return 0;
    }
    if (intensity == 0.0f || (red == 0.0f && green == 0.0f && blue == 0.0f)) {
        if (!bridge->screen_light.isNull()) {
            bridge->foreground_scene->remove(bridge->screen_light);
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
                    instance, intensity * kLegacyScreenCandelaScale *
                    kControllerScreenLightWeight);
            return 1;
        }
        bridge->foreground_scene->remove(bridge->screen_light);
        bridge->engine->destroy(bridge->screen_light);
        bridge->screen_light = {};
    }
    bridge->screen_light = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::FOCUSED_SPOT)
            .color(filament::LinearColor{red, green, blue})
            .intensityCandela(
                    intensity * kLegacyScreenCandelaScale *
                    kControllerScreenLightWeight)
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
    bridge->foreground_scene->addEntity(bridge->screen_light);
    return 1;
}

int bridge_screen_set_sampling(FilamentBridge* bridge, float filter_scale) {
    if (!bridge || !bridge->screen_material_instance ||
            !std::isfinite(filter_scale) || filter_scale < 1.0f ||
            filter_scale > 4.0f) {
        return 0;
    }
    bridge->screen_filter_scale = filter_scale;
    for (uint32_t eye_index = 0;
            eye_index < bridge->screen_mip_textures.size(); ++eye_index) {
        destroy_screen_mip_target(bridge, eye_index);
        destroy_screen_lanczos_target(bridge, eye_index);
    }
    for (auto* material_instance : bridge->screen_material_instances) {
        if (material_instance) {
            material_instance->setParameter("screenFilterScale", filter_scale);
        }
    }
    for (auto* material_instance : bridge->screen_mip_copy_material_instances) {
        if (material_instance) {
            material_instance->setParameter("screenFilterScale", filter_scale);
        }
    }
    return 1;
}

int bridge_screen_set_sampling_mode(FilamentBridge* bridge, int use_mip) {
    if (!bridge) return 0;
    bridge->screen_mip_experiment_enabled = use_mip != 0;
    if (bridge->active_eye < bridge->screen_textures.size()) {
        bind_screen_display_texture(bridge, bridge->active_eye);
    }
    return 1;
}

int bridge_screen_create(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine || !bridge->foreground_scene) return 0;
    bridge_screen_destroy(bridge);
    const char* eye_diagnostic_env = std::getenv("D2S_FILAMENT_EYE_DIAGNOSTIC");
    const bool eye_diagnostic = eye_diagnostic_env &&
            eye_diagnostic_env[0] != '\0' &&
            std::strcmp(eye_diagnostic_env, "0") != 0;
    // The screen sampler is used only for the internal MIP experiment texture.
    // External producer images use screen_source_texture_sampler below because
    // Filament external textures are restricted to LOD 0.
    bridge->screen_texture_sampler = filament::TextureSampler(
            filament::TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR,
            filament::TextureSampler::MagFilter::LINEAR,
            filament::TextureSampler::WrapMode::CLAMP_TO_EDGE);
    bridge->screen_source_texture_sampler = filament::TextureSampler(
            filament::TextureSampler::MinFilter::LINEAR,
            filament::TextureSampler::MagFilter::LINEAR,
            filament::TextureSampler::WrapMode::CLAMP_TO_EDGE);
    // Keep anisotropy on the real multi-level sampler. The source sampler is
    // intentionally left at ordinary linear filtering for external images.
    bridge->screen_texture_sampler.setAnisotropy(16.0f);
    const char* vertex_shader = R"FILAMENT(
        void materialVertex(inout MaterialVertexInputs material) {
            material.screenEyeIndex = vec4(float(getEyeIndex()), 0.0, 0.0, 0.0);
        }
    )FILAMENT";
    const char* shader = R"FILAMENT(
        float screen_sinc(float x) {
            x = abs(x);
            if (x < 0.00001) return 1.0;
            float p = 3.141592653589793 * x;
            return sin(p) / p;
        }

        float screen_lanczos2(float x) {
            x = abs(x);
            if (x >= 2.0) return 0.0;
            return screen_sinc(x) * screen_sinc(x * 0.5);
        }

        float screen_luma(vec3 color) {
            return color.b * 0.5 + (color.r * 0.5 + color.g);
        }

        vec3 screen_sample(vec2 uv, float lod) {
            if (variable_screenEyeIndex.x < 0.5) {
                return texture(materialParams_screenTextureLeft, uv, lod).rgb;
            }
            return texture(materialParams_screenTextureRight, uv, lod).rgb;
        }

        // Adapted from AMD FidelityFX FSR1 EASU (MIT). The scalar form keeps
        // the same edge direction, anisotropic lobe, deringing and 12 taps
        // while using Filament's material texture interface.
        vec3 screen_easu_source_sample(
                vec2 pixel, vec2 source_texel) {
            return screen_sample(
                    clamp((pixel + vec2(0.5)) * source_texel,
                            vec2(0.0), vec2(1.0)),
                    materialParams.screenLodBias);
        }

        void screen_easu_set(
                inout vec2 direction, inout float length_value,
                vec2 pp, float weight,
                float l_a, float l_b, float l_c, float l_d, float l_e) {
            float direction_x = l_d - l_b;
            float length_x = max(abs(l_d - l_c), abs(l_c - l_b));
            length_x = 1.0 / max(length_x, 0.000001);
            direction.x += direction_x * weight;
            length_x = clamp(abs(direction_x) * length_x, 0.0, 1.0);
            length_value += length_x * length_x * weight;

            float direction_y = l_e - l_a;
            float length_y = max(abs(l_e - l_c), abs(l_c - l_a));
            length_y = 1.0 / max(length_y, 0.000001);
            direction.y += direction_y * weight;
            length_y = clamp(abs(direction_y) * length_y, 0.0, 1.0);
            length_value += length_y * length_y * weight;
        }

        void screen_easu_tap(
                inout vec3 color, inout float weight_sum,
                vec2 offset, vec2 direction, vec2 length_value,
                float lobe, float clip_value, vec3 sample_color) {
            vec2 rotated = vec2(
                    offset.x * direction.x + offset.y * direction.y,
                    offset.x * -direction.y + offset.y * direction.x);
            rotated *= length_value;
            float distance_squared = min(dot(rotated, rotated), clip_value);
            float weight_b = 0.4 * distance_squared - 1.0;
            float weight_a = lobe * distance_squared - 1.0;
            weight_b *= weight_b;
            weight_a *= weight_a;
            weight_b = 1.5625 * weight_b - 0.5625;
            float weight = weight_b * weight_a;
            color += sample_color * weight;
            weight_sum += weight;
        }

        vec3 screen_easu(vec2 uv) {
            vec2 source_size = materialParams.screenSourceSize;
            vec2 source_texel = 1.0 / max(source_size, vec2(1.0));
            vec2 output_size = max(materialParams.screenOutputSize, vec2(1.0));
            vec2 output_uv = (floor(uv * output_size) + vec2(0.5)) / output_size;
            vec2 source_position = output_uv * source_size - vec2(0.5);
            vec2 source_base = floor(source_position);
            vec2 pp = source_position - source_base;

            vec3 b = screen_easu_source_sample(source_base + vec2(0.0, -1.0), source_texel);
            vec3 c = screen_easu_source_sample(source_base + vec2(1.0, -1.0), source_texel);
            vec3 e = screen_easu_source_sample(source_base + vec2(-1.0, 0.0), source_texel);
            vec3 f = screen_easu_source_sample(source_base + vec2(0.0, 0.0), source_texel);
            vec3 g = screen_easu_source_sample(source_base + vec2(1.0, 0.0), source_texel);
            vec3 h = screen_easu_source_sample(source_base + vec2(2.0, 0.0), source_texel);
            vec3 i = screen_easu_source_sample(source_base + vec2(-1.0, 1.0), source_texel);
            vec3 j = screen_easu_source_sample(source_base + vec2(0.0, 1.0), source_texel);
            vec3 k = screen_easu_source_sample(source_base + vec2(1.0, 1.0), source_texel);
            vec3 l = screen_easu_source_sample(source_base + vec2(2.0, 1.0), source_texel);
            vec3 n = screen_easu_source_sample(source_base + vec2(0.0, 2.0), source_texel);
            vec3 o = screen_easu_source_sample(source_base + vec2(1.0, 2.0), source_texel);

            float b_luma = screen_luma(b);
            float c_luma = screen_luma(c);
            float e_luma = screen_luma(e);
            float f_luma = screen_luma(f);
            float g_luma = screen_luma(g);
            float h_luma = screen_luma(h);
            float i_luma = screen_luma(i);
            float j_luma = screen_luma(j);
            float k_luma = screen_luma(k);
            float l_luma = screen_luma(l);
            float n_luma = screen_luma(n);
            float o_luma = screen_luma(o);

            vec2 direction = vec2(0.0);
            float length_value = 0.0;
            screen_easu_set(direction, length_value, pp,
                    (1.0 - pp.x) * (1.0 - pp.y),
                    b_luma, e_luma, f_luma, g_luma, j_luma);
            screen_easu_set(direction, length_value, pp,
                    pp.x * (1.0 - pp.y),
                    c_luma, f_luma, g_luma, h_luma, k_luma);
            screen_easu_set(direction, length_value, pp,
                    (1.0 - pp.x) * pp.y,
                    f_luma, i_luma, j_luma, k_luma, n_luma);
            screen_easu_set(direction, length_value, pp,
                    pp.x * pp.y,
                    g_luma, j_luma, k_luma, l_luma, o_luma);

            float direction_length = dot(direction, direction);
            bool zero_direction = direction_length < 0.000030517578125;
            direction_length = zero_direction
                    ? 1.0 : 1.0 / sqrt(direction_length);
            direction = zero_direction
                    ? vec2(1.0, 0.0) : direction * direction_length;
            length_value = length_value * 0.5;
            length_value *= length_value;
            float stretch = dot(direction, direction) /
                    max(max(abs(direction.x), abs(direction.y)), 0.000001);
            vec2 length_squared = vec2(
                    1.0 + (stretch - 1.0) * length_value,
                    1.0 - 0.5 * length_value);
            float lobe = 0.5 + (0.21 - 0.5) * length_value;
            float clip_value = 1.0 / max(lobe, 0.000001);

            vec3 min4 = min(min(f, g), min(j, k));
            vec3 max4 = max(max(f, g), max(j, k));
            vec3 color = vec3(0.0);
            float weight_sum = 0.0;
            screen_easu_tap(color, weight_sum, vec2(0.0, -1.0) - pp,
                    direction, length_squared, lobe, clip_value, b);
            screen_easu_tap(color, weight_sum, vec2(1.0, -1.0) - pp,
                    direction, length_squared, lobe, clip_value, c);
            screen_easu_tap(color, weight_sum, vec2(-1.0, 1.0) - pp,
                    direction, length_squared, lobe, clip_value, i);
            screen_easu_tap(color, weight_sum, vec2(0.0, 1.0) - pp,
                    direction, length_squared, lobe, clip_value, j);
            screen_easu_tap(color, weight_sum, vec2(0.0, 0.0) - pp,
                    direction, length_squared, lobe, clip_value, f);
            screen_easu_tap(color, weight_sum, vec2(-1.0, 0.0) - pp,
                    direction, length_squared, lobe, clip_value, e);
            screen_easu_tap(color, weight_sum, vec2(1.0, 1.0) - pp,
                    direction, length_squared, lobe, clip_value, k);
            screen_easu_tap(color, weight_sum, vec2(2.0, 1.0) - pp,
                    direction, length_squared, lobe, clip_value, l);
            screen_easu_tap(color, weight_sum, vec2(2.0, 0.0) - pp,
                    direction, length_squared, lobe, clip_value, h);
            screen_easu_tap(color, weight_sum, vec2(1.0, 0.0) - pp,
                    direction, length_squared, lobe, clip_value, g);
            screen_easu_tap(color, weight_sum, vec2(1.0, 2.0) - pp,
                    direction, length_squared, lobe, clip_value, o);
            screen_easu_tap(color, weight_sum, vec2(0.0, 2.0) - pp,
                    direction, length_squared, lobe, clip_value, n);
            if (weight_sum <= 0.000001) return f;
            return min(max4, max(min4, color / weight_sum));
        }

        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            if (materialParams.screenEyeDiagnostic > 0.5) {
                material.baseColor = variable_screenEyeIndex.x < 0.5
                        ? vec4(1.0, 0.0, 0.0, 1.0)
                        : vec4(0.0, 1.0, 0.0, 1.0);
                return;
            }
            vec2 uv = getUV0();
            vec2 texel = materialParams.screenTexelSize;
            vec3 center = screen_sample(uv, materialParams.screenLodBias);
            vec3 output_color = center;
            float quality_pass = materialParams.screenQualityPass;

            if (quality_pass > 0.5 && quality_pass < 1.5) {
                // Legacy screen-quality pass 1: separable-equivalent 4x4
                // Lanczos2 reconstruction performed entirely on the GPU.
                vec2 source_position = uv / texel - vec2(0.5);
                vec2 source_base = floor(source_position);
                vec3 accum = vec3(0.0);
                float weight_sum = 0.0;
                for (int y = -1; y <= 2; ++y) {
                    for (int x = -1; x <= 2; ++x) {
                        vec2 sample_position = source_base + vec2(float(x), float(y));
                        vec2 delta = source_position - sample_position;
                        float weight = screen_lanczos2(delta.x) *
                                screen_lanczos2(delta.y);
                        vec2 sample_uv = (sample_position + vec2(0.5)) * texel;
                        accum += screen_sample(
                                clamp(sample_uv, vec2(0.0), vec2(1.0)),
                                materialParams.screenLodBias) * weight;
                        weight_sum += weight;
                    }
                }
                output_color = accum / max(weight_sum, 0.000001);
            } else if (quality_pass > 2.5 && quality_pass < 3.5) {
                output_color = screen_easu(uv);
            } else if ((quality_pass > 1.5 && quality_pass < 2.5) ||
                    quality_pass > 3.5) {
                // Legacy screen-quality pass 2: full FSR RCAS. This is the
                // same cross-shaped luma adaptation and RGB limiter used by
                // the legacy renderer, evaluated in the GPU pass before MIP.
                vec2 north_uv = clamp(uv + vec2(0.0, -texel.y),
                        vec2(0.0), vec2(1.0));
                vec2 west_uv = clamp(uv + vec2(-texel.x, 0.0),
                        vec2(0.0), vec2(1.0));
                vec2 east_uv = clamp(uv + vec2(texel.x, 0.0),
                        vec2(0.0), vec2(1.0));
                vec2 south_uv = clamp(uv + vec2(0.0, texel.y),
                        vec2(0.0), vec2(1.0));
                vec3 b = screen_sample(north_uv, materialParams.screenLodBias);
                vec3 d = screen_sample(west_uv, materialParams.screenLodBias);
                vec3 e = center;
                vec3 f = screen_sample(east_uv, materialParams.screenLodBias);
                vec3 h = screen_sample(south_uv, materialParams.screenLodBias);
                float b_luma = screen_luma(b);
                float d_luma = screen_luma(d);
                float e_luma = screen_luma(e);
                float f_luma = screen_luma(f);
                float h_luma = screen_luma(h);
                float nz = 0.25 * b_luma + 0.25 * d_luma +
                        0.25 * f_luma + 0.25 * h_luma - e_luma;
                float l_max = max(max(max(b_luma, d_luma),
                        max(e_luma, f_luma)), h_luma);
                float l_min = min(min(min(b_luma, d_luma),
                        min(e_luma, f_luma)), h_luma);
                nz = clamp(abs(nz) / max(abs(l_max - l_min), 0.000001),
                        0.0, 1.0);
                nz = -0.5 * nz + 1.0;
                vec3 min4 = min(min(b, d), min(f, h));
                vec3 max4 = max(max(b, d), max(f, h));
                vec3 hit_min = min(min4, e) /
                        max(4.0 * max4, vec3(0.000001));
                vec3 hit_max = (vec3(1.0) - max(max4, e)) /
                        min(4.0 * min4 - vec3(4.0), vec3(-0.000001));
                vec3 lobe_rgb = max(-hit_min, hit_max);
                float lobe = max(max(lobe_rgb.r, lobe_rgb.g), lobe_rgb.b);
                float rcas_limit = 0.25 - (1.0 / 16.0);
                float sharpness = clamp(
                        materialParams.screenSharpness /
                                max(materialParams.screenFilterScale, 1.0),
                        0.0, 1.0);
                float sharpness_stops = 2.0 * (1.0 - sharpness);
                float contrast = exp2(-sharpness_stops);
                lobe = max(-rcas_limit, min(lobe, 0.0)) * contrast * nz;
                float reciprocal_lobe = 1.0 / max(abs(4.0 * lobe + 1.0),
                        0.000001);
                output_color = clamp((lobe * b + lobe * d + lobe * h +
                        lobe * f + e) * reciprocal_lobe,
                        vec3(0.0), vec3(1.0));
            }
            // Scene exposure is a view-wide operation. Cancel it for the
            // display-referred screen so desktop sRGB bytes are not brightened
            // together with the HDR environment.
            output_color *= materialParams.screenExposureCompensation;
            material.baseColor = vec4(clamp(output_color,
                    vec3(0.0), vec3(1.0)), 1.0);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S OpenXR Screen")
            .material(shader)
            .materialVertex(vertex_shader)
            .variable(filamat::MaterialBuilder::Variable::CUSTOM0, "screenEyeIndex")
            .require(filament::VertexAttribute::UV0)
            .parameter("screenTextureLeft", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("screenTextureRight", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("screenTexelSize", filamat::MaterialBuilder::UniformType::FLOAT2)
            .parameter("screenSourceSize", filamat::MaterialBuilder::UniformType::FLOAT2)
            .parameter("screenOutputSize", filamat::MaterialBuilder::UniformType::FLOAT2)
            .parameter("screenFilterScale", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenSharpness", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenLodBias", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenExposureCompensation", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenQualityPass", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenEyeDiagnostic", filamat::MaterialBuilder::UniformType::FLOAT)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            // Match the legacy projection pass: the display is an opaque
            // image, but it must not write depth. Otherwise a screen in front
            // of the hands can occlude controllers regardless of the
            // controller View's own depth clear behavior.
            .blending(filament::BlendingMode::OPAQUE)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false)
            .depthCulling(true)
            .stereoscopicType(filamat::MaterialBuilder::StereoscopicType::MULTIVIEW)
            .stereoscopicEyeCount(2)
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
    for (auto*& material_instance : bridge->screen_material_instances) {
        material_instance = bridge->screen_material->createInstance();
        if (!material_instance) {
            bridge_set_error(bridge,
                    "Filament could not create per-eye screen material");
            return 0;
        }
        material_instance->setParameter(
                "screenFilterScale", bridge->screen_filter_scale);
        material_instance->setParameter(
                "screenSourceSize", filament::math::float2{1.0f, 1.0f});
        material_instance->setParameter(
                "screenOutputSize", filament::math::float2{1.0f, 1.0f});
        material_instance->setParameter(
                "screenSharpness", bridge->screen_filter_sharpness);
        material_instance->setParameter("screenLodBias", kLegacyScreenLodBias);
        material_instance->setParameter("screenExposureCompensation", 1.0f);
        material_instance->setParameter("screenQualityPass", 0.0f);
        material_instance->setParameter(
                "screenEyeDiagnostic", eye_diagnostic ? 1.0f : 0.0f);
    }
    if (eye_diagnostic) {
        std::fprintf(stderr,
                "[FilamentBridge] eye diagnostic enabled: left=red right=green\n");
        std::fflush(stderr);
    }
    bridge->screen_material_instance =
            bridge->screen_material_instances[bridge->active_eye];
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
    for (uint32_t eye_index = 0; eye_index < bridge->screen_entities.size(); ++eye_index) {
        auto& entity = bridge->screen_entities[eye_index];
        entity = utils::EntityManager::get().create();
        const auto result = filament::RenderableManager::Builder(1)
                .boundingBox({{-20000.0f, -20000.0f, -20000.0f}, {20000.0f, 20000.0f, 20000.0f}})
                .material(0, bridge->screen_material_instances[eye_index])
                .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                        bridge->screen_vertex_buffer, bridge->screen_index_buffer,
                        0, static_cast<uint32_t>(bridge->screen_indices.size()))
                // The surround shell is the background effect (priority 0), then
                // the display is drawn, followed by foreground edge effects.
                .priority(2).culling(false).castShadows(false).receiveShadows(false)
                .build(*bridge->engine, entity);
        if (result != filament::RenderableManager::Builder::Success) {
            bridge_set_error(bridge, "Filament could not create OpenXR screen renderable");
            return 0;
        }
        bridge_set_renderable_layer(
                bridge, entity, kScreenLayerBase + eye_index, false);
    }
    bridge->screen_entity = bridge->screen_entities[bridge->active_eye];
    // Display-referred screen content bypasses the HDR scene view.
    for (auto*& material_instance :
            bridge->screen_mip_copy_material_instances) {
        material_instance = bridge->screen_material->createInstance();
        if (!material_instance) {
            bridge_set_error(bridge,
                    "Filament could not create per-eye screen copy material");
            return 0;
        }
        material_instance->setParameter(
                "screenFilterScale", bridge->screen_filter_scale);
        material_instance->setParameter(
                "screenSourceSize", filament::math::float2{1.0f, 1.0f});
        material_instance->setParameter(
                "screenOutputSize", filament::math::float2{1.0f, 1.0f});
        material_instance->setParameter(
                "screenSharpness", bridge->screen_filter_sharpness);
        material_instance->setParameter("screenLodBias", kLegacyScreenLodBias);
        // The copy view is only a linear/sRGB-preserving filter pass. It must
        // not inherit the scene exposure compensation used by the final draw.
        material_instance->setParameter("screenExposureCompensation", 1.0f);
        material_instance->setParameter("screenQualityPass", 0.0f);
    }
    bridge->screen_mip_copy_material_instance =
            bridge->screen_mip_copy_material_instances[bridge->active_eye];
    // Match the legacy screen mesh orientation: v=0 is the bottom edge and
    // v=1 is the top edge. Reversing this pair mirrors the copied image.
    bridge->screen_mip_copy_vertices = {
            PreviewScreenVertex{{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            PreviewScreenVertex{{-1.0f,  1.0f, 0.0f}, {0.0f, 1.0f}},
            PreviewScreenVertex{{ 1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            PreviewScreenVertex{{ 1.0f,  1.0f, 0.0f}, {1.0f, 1.0f}},
    };
    bridge->screen_mip_copy_indices = {0, 2, 1, 2, 3, 1};
    bridge->screen_mip_copy_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(static_cast<uint32_t>(bridge->screen_mip_copy_vertices.size()))
            .bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(PreviewScreenVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(PreviewScreenVertex))
            .build(*bridge->engine);
    bridge->screen_mip_copy_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(static_cast<uint32_t>(bridge->screen_mip_copy_indices.size()))
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*bridge->engine);
    if (!bridge->screen_mip_copy_material_instance ||
            !bridge->screen_mip_copy_vertex_buffer ||
            !bridge->screen_mip_copy_index_buffer) {
        bridge_set_error(bridge, "Filament could not create screen MIP copy resources");
        return 0;
    }
    bridge->screen_mip_copy_vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    bridge->screen_mip_copy_vertices.data(),
                    bridge->screen_mip_copy_vertices.size() * sizeof(PreviewScreenVertex), nullptr));
    bridge->screen_mip_copy_index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    bridge->screen_mip_copy_indices.data(),
                    bridge->screen_mip_copy_indices.size() * sizeof(uint16_t), nullptr));
    for (uint32_t eye_index = 0;
            eye_index < bridge->screen_mip_copy_entities.size(); ++eye_index) {
        auto& entity = bridge->screen_mip_copy_entities[eye_index];
        entity = utils::EntityManager::get().create();
        const auto copy_result = filament::RenderableManager::Builder(1)
                .boundingBox({{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}})
                .material(0, bridge->screen_mip_copy_material_instances[eye_index])
                .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                        bridge->screen_mip_copy_vertex_buffer,
                        bridge->screen_mip_copy_index_buffer,
                        0, static_cast<uint32_t>(bridge->screen_mip_copy_indices.size()))
                .priority(0).culling(false).castShadows(false).receiveShadows(false)
                .build(*bridge->engine, entity);
        if (copy_result != filament::RenderableManager::Builder::Success) {
            bridge_set_error(bridge, "Filament could not create screen MIP copy renderable");
            return 0;
        }
        bridge->scene->addEntity(entity);
        bridge_set_renderable_layer(
                bridge, entity, kScreenMipCopyLayerBase + eye_index, true);
    }
    bridge->screen_mip_copy_entity =
            bridge->screen_mip_copy_entities[bridge->active_eye];
    bridge->screen_mip_copy_camera = bridge->engine->createCamera(
            utils::EntityManager::get().create());
    if (!bridge->screen_mip_copy_camera) {
        bridge_set_error(bridge, "Filament could not create screen MIP copy view");
        return 0;
    }
    bridge->screen_mip_copy_camera->setProjection(
            filament::Camera::Projection::ORTHO,
            -1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
    bridge->screen_mip_copy_camera->lookAt(
            filament::math::float3{0.0f, 0.0f, 1.0f},
            filament::math::float3{0.0f, 0.0f, 0.0f},
            filament::math::float3{0.0f, 1.0f, 0.0f});
    for (uint32_t eye_index = 0;
            eye_index < bridge->screen_mip_copy_views.size(); ++eye_index) {
        auto*& view = bridge->screen_mip_copy_views[eye_index];
        view = bridge->engine->createView();
        if (!view) {
            bridge_set_error(bridge, "Filament could not create screen MIP copy view");
            return 0;
        }
        view->setScene(bridge->scene);
        view->setCamera(bridge->screen_mip_copy_camera);
        view->setVisibleLayers(
                0xff, static_cast<uint8_t>(
                        1u << (kScreenMipCopyLayerBase + eye_index)));
        view->setPostProcessingEnabled(false);
        view->setAntiAliasing(filament::AntiAliasing::NONE);
    }
    bridge->screen_mip_copy_view =
            bridge->screen_mip_copy_views[bridge->active_eye];
    // The sampler is required by the material. Keep the renderable detached
    // until a valid runtime Vulkan image has been imported.
    if (!bridge_glow_create(bridge)) {
        return 0;
    }
    return 1;
}

int bridge_screen_set_image(FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->screen_material_instance ||
            !image || width == 0 || height == 0) return 0;
    if (format != VK_FORMAT_R8G8B8A8_SRGB && format != VK_FORMAT_R8G8B8A8_UNORM) {
        bridge_set_error(bridge,
                "Virtual screen requires VK_FORMAT_R8G8B8A8_SRGB or VK_FORMAT_R8G8B8A8_UNORM");
        return 0;
    }
    const uint32_t eye_index = bridge->active_eye;
    bridge->screen_source_formats[eye_index] = format;
    for (const auto& slot : bridge->screen_texture_cache[eye_index]) {
        if (slot.image == image && slot.width == width &&
                slot.height == height && slot.format == format && slot.texture) {
            bridge->screen_source_textures[eye_index] = slot.texture;
            bind_screen_copy_source(bridge, slot.texture, width, height);
            ensure_screen_quality_targets(bridge, eye_index, width, height, format);
            bind_screen_display_texture(bridge, eye_index);
            ++bridge->screen_source_bind_count[eye_index];
            add_screen_entities(bridge);
            return 1;
        }
    }
#if defined(D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE)
    const auto external_image =
            bridge->platform->createExternalImageFromVkImage(
                    reinterpret_cast<VkImage>(const_cast<void*>(image)),
                    static_cast<VkFormat>(format), width, height);
    if (!external_image) {
        bridge_set_error(bridge,
                "Filament Vulkan backend rejected the external VkImage metadata");
        return 0;
    }
    const auto texture_format = format == VK_FORMAT_R8G8B8A8_SRGB
            ? filament::Texture::InternalFormat::SRGB8_A8
            : filament::Texture::InternalFormat::RGBA8;
    auto* texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            .format(texture_format)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .usage(filament::Texture::Usage::SAMPLEABLE)
            .external()
            .build(*bridge->engine);
    if (texture) {
        texture->setExternalImage(*bridge->engine, external_image);
    }
#else
    auto* texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            // Runtime eye images contain display-referred sRGB bytes in a
            // Vulkan SRGB image; decode them exactly once on sample.
            .format(filament::Texture::InternalFormat::SRGB8_A8)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .usage(filament::Texture::Usage::SAMPLEABLE)
            .import(reinterpret_cast<intptr_t>(const_cast<void*>(image)))
            .build(*bridge->engine);
#endif
    if (!texture) {
        bridge_set_error(bridge, "Filament could not import virtual screen Vulkan image");
        return 0;
    }
    bridge->screen_texture_cache[eye_index].push_back(
            ScreenTextureSlot{image, texture, width, height, format});
    bridge->screen_source_textures[eye_index] = texture;
    bind_screen_copy_source(bridge, texture, width, height);
    ensure_screen_quality_targets(bridge, eye_index, width, height, format);
    bind_screen_display_texture(bridge, eye_index);
    ++bridge->screen_source_bind_count[eye_index];
    add_screen_entities(bridge);
    return 1;
}

int bridge_screen_set_source_version(
        FilamentBridge* bridge, uint64_t version) {
    if (!bridge || bridge->active_eye >= bridge->screen_source_versions.size()) {
        return 0;
    }
    bridge->screen_source_versions[bridge->active_eye] = version;
    return 1;
}

int bridge_screen_prepare_eye(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || !bridge->frame_active || !bridge->renderer ||
            !bridge->screen_mip_experiment_enabled ||
            eye_index >= bridge->screen_mip_textures.size() ||
            !bridge->screen_source_textures[eye_index] ||
            !bridge->screen_mip_copy_views[eye_index] ||
            !bridge->screen_mip_copy_material_instances[eye_index]) {
        return 0;
    }
    if (bridge->screen_mip_ready[eye_index] &&
            bridge->screen_source_versions[eye_index] ==
                    bridge->screen_last_mip_versions[eye_index]) {
        return 1;
    }
    auto* source = bridge->screen_source_textures[eye_index];
    if (!ensure_screen_quality_targets(
            bridge, eye_index, source->getWidth(), source->getHeight(),
            bridge->screen_source_formats[eye_index])) {
        return 0;
    }
    auto* material = bridge->screen_mip_copy_material_instances[eye_index];
    auto* view = bridge->screen_mip_copy_views[eye_index];
    auto* lanczos_texture = bridge->screen_lanczos_textures[eye_index];
    auto* mip_texture = bridge->screen_mip_textures[eye_index];
    auto* lanczos_target = bridge->screen_lanczos_render_targets[eye_index];
    auto* mip_target = bridge->screen_mip_render_targets[eye_index];

    const auto bind_source = [&](filament::Texture* texture,
                                 const filament::TextureSampler& sampler) {
        material->setParameter("screenTextureLeft", texture, sampler);
        material->setParameter("screenTextureRight", texture, sampler);
        material->setParameter("screenTexelSize", filament::math::float2{
                1.0f / static_cast<float>(texture->getWidth()),
                1.0f / static_cast<float>(texture->getHeight())});
        material->setParameter("screenSourceSize", filament::math::float2{
                static_cast<float>(texture->getWidth()),
                static_cast<float>(texture->getHeight())});
    };
    const auto set_output_size = [&](filament::Texture* texture) {
        material->setParameter("screenOutputSize", filament::math::float2{
                static_cast<float>(texture->getWidth()),
                static_cast<float>(texture->getHeight())});
    };

    // Match the legacy OpenGL default: a matched input/headset tier must not
    // pass through an additional Lanczos2 + RCAS filter. Keep the dynamic MIP
    // chain, but copy the source directly into LOD 0 first.
    if (bridge->screen_filter_scale <= 1.0001f &&
            bridge->screen_upscale_scale <= 1.0001f) {
        bind_source(source, bridge->screen_source_texture_sampler);
        material->setParameter("screenQualityPass", 0.0f);
        set_output_size(mip_texture);
        view->setRenderTarget(mip_target);
        view->setViewport(filament::Viewport{
                0, 0, static_cast<uint32_t>(mip_texture->getWidth()),
                static_cast<uint32_t>(mip_texture->getHeight())});
        bridge->renderer->render(view);
        mip_texture->generateMipmaps(*bridge->engine);
        ++bridge->screen_mip_generation_count[eye_index];
        bridge->screen_last_mip_versions[eye_index] =
                bridge->screen_source_versions[eye_index];
        bridge->screen_mip_ready[eye_index] = true;
        return 1;
    }

    const bool use_easu = bridge->screen_upscale_scale > 1.0001f;
    // Pass 1: reconstruct the external source into the selected quality target.
    bind_source(source, bridge->screen_source_texture_sampler);
    material->setParameter("screenQualityPass", use_easu ? 3.0f : 1.0f);
    set_output_size(lanczos_texture);
    view->setRenderTarget(lanczos_target);
    view->setViewport(filament::Viewport{
            0, 0, static_cast<uint32_t>(lanczos_texture->getWidth()),
            static_cast<uint32_t>(lanczos_texture->getHeight())});
    bridge->renderer->render(view);

    // Pass 2: sharpen the reconstructed image, then generate the stable MIP
    // chain from the final quality result for video minification.
    bind_source(lanczos_texture, bridge->screen_source_texture_sampler);
    material->setParameter("screenQualityPass", use_easu ? 4.0f : 2.0f);
    set_output_size(mip_texture);
    view->setRenderTarget(mip_target);
    view->setViewport(filament::Viewport{
            0, 0, static_cast<uint32_t>(mip_texture->getWidth()),
            static_cast<uint32_t>(mip_texture->getHeight())});
    bridge->renderer->render(view);
    mip_texture->generateMipmaps(*bridge->engine);
    ++bridge->screen_mip_generation_count[eye_index];
    bridge->screen_last_mip_versions[eye_index] =
            bridge->screen_source_versions[eye_index];
    bridge->screen_mip_ready[eye_index] = true;
    return 1;
}

int bridge_screen_set_upscale(FilamentBridge* bridge, float upscale_scale) {
    if (!bridge || !bridge->screen_material_instance ||
            !std::isfinite(upscale_scale) || upscale_scale < 1.0f ||
            upscale_scale > 4.0f) {
        return 0;
    }
    bridge->screen_upscale_scale = upscale_scale;
    // The source image may already be imported when the Python policy is
    // applied. Recreate targets lazily with the new output extent on the next
    // frame instead of sampling a target allocated for the old scale.
    for (uint32_t eye_index = 0;
            eye_index < bridge->screen_mip_textures.size(); ++eye_index) {
        destroy_screen_mip_target(bridge, eye_index);
        destroy_screen_lanczos_target(bridge, eye_index);
    }
    return 1;
}

int bridge_screen_prepare_frame(FilamentBridge* bridge) {
    return bridge ? bridge_screen_prepare_eye(bridge, bridge->active_eye) : 0;
}

int bridge_screen_bind_stereo_textures(FilamentBridge* bridge) {
    if (!bridge || !bridge->multiview_active ||
            !bridge->screen_material_instances[0]) return 0;
    bind_screen_display_texture(bridge, 0);
    bind_screen_display_texture(bridge, 1);
    auto* left = bridge->screen_textures[0];
    auto* right = bridge->screen_textures[1];
    if (!left || !right) return 0;
    auto* material = bridge->screen_material_instances[0];
    material->setParameter(
            "screenTextureLeft", left,
            bridge->screen_mip_ready[0] ? bridge->screen_texture_sampler
                                        : bridge->screen_source_texture_sampler);
    material->setParameter(
            "screenTextureRight", right,
            bridge->screen_mip_ready[1] ? bridge->screen_texture_sampler
                                        : bridge->screen_source_texture_sampler);
    material->setParameter("screenTexelSize", filament::math::float2{
            1.0f / static_cast<float>(left->getWidth()),
            1.0f / static_cast<float>(left->getHeight())});
    return 1;
}

int bridge_screen_set_fixed_image(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height) {
    if (!bridge || !bridge->engine || !rgba || width == 0 || height == 0 ||
            !bridge->screen_material_instance ||
            !bridge->screen_mip_copy_material_instance) {
        return 0;
    }
    if (bridge->screen_fixed_source_texture) {
        bridge->engine->destroy(bridge->screen_fixed_source_texture);
        bridge->screen_fixed_source_texture = nullptr;
    }
    auto* texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            .format(filament::Texture::InternalFormat::SRGB8_A8)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            // setImage() requires UPLOADABLE; DEFAULT includes both upload
            // and sampling usage for this immutable regression source.
            .usage(filament::Texture::Usage::DEFAULT)
            .build(*bridge->engine);
    if (!texture) {
        bridge_set_error(bridge, "Filament could not create fixed screen texture");
        return 0;
    }
    const size_t byte_count = static_cast<size_t>(width) * height * 4u;
    auto* pixels = new uint8_t[byte_count];
    std::memcpy(pixels, rgba, byte_count);
    filament::Texture::PixelBufferDescriptor descriptor(
            pixels, byte_count, filament::Texture::Format::RGBA,
            filament::Texture::Type::UBYTE,
            [](void* buffer, size_t, void*) {
                delete[] static_cast<uint8_t*>(buffer);
            });
    texture->setImage(*bridge->engine, 0, std::move(descriptor));
    bridge->screen_fixed_source_texture = texture;
    for (uint32_t eye_index = 0;
            eye_index < bridge->screen_source_textures.size(); ++eye_index) {
        bridge->screen_source_textures[eye_index] = texture;
        bridge->screen_source_formats[eye_index] = VK_FORMAT_R8G8B8A8_SRGB;
        ensure_screen_quality_targets(bridge, eye_index, width, height,
                VK_FORMAT_R8G8B8A8_SRGB);
    }
    bind_screen_copy_source(bridge, texture, width, height);
    bind_screen_display_texture(bridge, bridge->active_eye);
    add_screen_entities(bridge);
    return 1;
}

int bridge_screen_get_sampling_stats(
        FilamentBridge* bridge, uint32_t eye_index,
        uint64_t* source_binds, uint64_t* mip_generations) {
    if (!bridge || eye_index >= bridge->screen_source_bind_count.size() ||
            !source_binds || !mip_generations) {
        return 0;
    }
    *source_binds = bridge->screen_source_bind_count[eye_index];
    *mip_generations = bridge->screen_mip_generation_count[eye_index];
    return 1;
}
