#include "bridge_screen.h"
#include "bridge_internal.h"

#include <backend/PixelBufferDescriptor.h>

#include <cstring>

namespace {

constexpr uint32_t kScreenSegments = 48;
constexpr float kCurvedHalfAngle = 0.72f;
constexpr float kLegacyScreenCandelaScale = 1200.0f;
constexpr uint8_t kScreenMipCopyLayer = 2;

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
    if (texture && bridge->screen_material_instance) {
        bridge->screen_material_instance->setParameter(
                "screenTexture", texture,
                use_mip ? bridge->screen_texture_sampler
                        : bridge->screen_source_texture_sampler);
        bridge->screen_material_instance->setParameter(
                "screenTexelSize", filament::math::float2{
                        1.0f / static_cast<float>(texture->getWidth()),
                        1.0f / static_cast<float>(texture->getHeight())});
    }
}

void bind_screen_copy_source(
        FilamentBridge* bridge, filament::Texture* source,
        uint32_t width, uint32_t height) {
    if (!bridge || !source || !bridge->screen_mip_copy_material_instance) return;
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexture", source, bridge->screen_source_texture_sampler);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexelSize", filament::math::float2{
                    1.0f / static_cast<float>(width),
                    1.0f / static_cast<float>(height)});
}

bool ensure_screen_mip_target(
        FilamentBridge* bridge, uint32_t eye_index,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->screen_mip_experiment_enabled ||
            !bridge->screen_mip_copy_view || bridge->screen_mip_copy_entity.isNull() ||
            eye_index >= bridge->screen_mip_textures.size() ||
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
    bridge->screen_mip_ready[eye_index] = true;
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
    const float scale = std::max(1.0f, bridge->screen_filter_scale);
    const uint32_t width = std::max(
            16u, static_cast<uint32_t>(std::lround(source_width / scale)) & ~1u);
    const uint32_t height = std::max(
            16u, static_cast<uint32_t>(std::lround(source_height / scale)) & ~1u);
    return ensure_screen_lanczos_target(bridge, eye_index, width, height, format) &&
            ensure_screen_mip_target(bridge, eye_index, width, height, format);
}

}  // namespace

void bridge_screen_destroy(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    if (bridge->scene && !bridge->screen_mip_copy_entity.isNull()) {
        bridge->scene->remove(bridge->screen_mip_copy_entity);
    }
    if (!bridge->screen_mip_copy_entity.isNull()) {
        bridge->engine->destroy(bridge->screen_mip_copy_entity);
        bridge->screen_mip_copy_entity = {};
    }
    if (bridge->screen_mip_copy_view) {
        bridge->engine->destroy(bridge->screen_mip_copy_view);
        bridge->screen_mip_copy_view = nullptr;
    }
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
    if (bridge->screen_mip_copy_material_instance) {
        bridge->engine->destroy(bridge->screen_mip_copy_material_instance);
        bridge->screen_mip_copy_material_instance = nullptr;
    }
    if (bridge->screen_fixed_source_texture) {
        bridge->engine->destroy(bridge->screen_fixed_source_texture);
        bridge->screen_fixed_source_texture = nullptr;
    }
    for (uint32_t eye_index = 0; eye_index < bridge->screen_mip_textures.size(); ++eye_index) {
        destroy_screen_lanczos_target(bridge, eye_index);
        destroy_screen_mip_target(bridge, eye_index);
    }
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

int bridge_screen_set_sampling(FilamentBridge* bridge, float filter_scale) {
    if (!bridge || !bridge->screen_material_instance ||
            !std::isfinite(filter_scale) || filter_scale < 1.0f ||
            filter_scale > 4.0f) {
        return 0;
    }
    bridge->screen_filter_scale = filter_scale;
    bridge->screen_material_instance->setParameter(
            "screenFilterScale", filter_scale);
    if (bridge->screen_mip_copy_material_instance) {
        bridge->screen_mip_copy_material_instance->setParameter(
                "screenFilterScale", filter_scale);
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
    if (!bridge || !bridge->engine || !bridge->scene) return 0;
    bridge_screen_destroy(bridge);
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

        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            vec2 uv = getUV0();
            vec2 texel = materialParams_screenTexelSize;
            vec3 center = texture(materialParams_screenTexture, uv).rgb;
            vec3 output_color = center;
            float quality_pass = materialParams_screenQualityPass;

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
                        accum += texture(materialParams_screenTexture,
                                clamp(sample_uv, vec2(0.0), vec2(1.0))).rgb * weight;
                        weight_sum += weight;
                    }
                }
                output_color = accum / max(weight_sum, 0.000001);
            } else if (quality_pass > 1.5) {
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
                vec3 b = texture(materialParams_screenTexture, north_uv).rgb;
                vec3 d = texture(materialParams_screenTexture, west_uv).rgb;
                vec3 e = center;
                vec3 f = texture(materialParams_screenTexture, east_uv).rgb;
                vec3 h = texture(materialParams_screenTexture, south_uv).rgb;
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
                        materialParams_screenSharpness /
                                max(materialParams_screenFilterScale, 1.0),
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
            material.baseColor = vec4(clamp(output_color,
                    vec3(0.0), vec3(1.0)), 1.0);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S OpenXR Screen")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .parameter("screenTexture", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("screenTexelSize", filamat::MaterialBuilder::UniformType::FLOAT2)
            .parameter("screenFilterScale", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenSharpness", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("screenQualityPass", filamat::MaterialBuilder::UniformType::FLOAT)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            // Match the legacy projection pass: the display is an opaque
            // image that writes depth before foreground controllers render.
            .blending(filament::BlendingMode::OPAQUE)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(true)
            .depthCulling(true)
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
    bridge->screen_material_instance->setParameter(
            "screenFilterScale", bridge->screen_filter_scale);
    bridge->screen_material_instance->setParameter(
            "screenSharpness", bridge->screen_filter_sharpness);
    bridge->screen_material_instance->setParameter("screenQualityPass", 0.0f);
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
            // Render the display-referred screen before controllers and laser,
            // matching the legacy projection pass and preserving depth order.
            .priority(0).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, bridge->screen_entity);
    if (result != filament::RenderableManager::Builder::Success) {
        bridge_set_error(bridge, "Filament could not create OpenXR screen renderable");
        return 0;
    }
    // Display-referred screen content bypasses the HDR scene view.
    bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, false);
    bridge->screen_mip_copy_material_instance =
            bridge->screen_material->createInstance();
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenFilterScale", bridge->screen_filter_scale);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenSharpness", bridge->screen_filter_sharpness);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenQualityPass", 0.0f);
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
    bridge->screen_mip_copy_entity = utils::EntityManager::get().create();
    const auto copy_result = filament::RenderableManager::Builder(1)
            .boundingBox({{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}})
            .material(0, bridge->screen_mip_copy_material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    bridge->screen_mip_copy_vertex_buffer,
                    bridge->screen_mip_copy_index_buffer,
                    0, static_cast<uint32_t>(bridge->screen_mip_copy_indices.size()))
            .priority(0).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, bridge->screen_mip_copy_entity);
    if (copy_result != filament::RenderableManager::Builder::Success) {
        bridge_set_error(bridge, "Filament could not create screen MIP copy renderable");
        return 0;
    }
    bridge->scene->addEntity(bridge->screen_mip_copy_entity);
    bridge_set_renderable_layer(
            bridge, bridge->screen_mip_copy_entity, kScreenMipCopyLayer, true);
    bridge->screen_mip_copy_camera = bridge->engine->createCamera(
            utils::EntityManager::get().create());
    bridge->screen_mip_copy_view = bridge->engine->createView();
    if (!bridge->screen_mip_copy_camera || !bridge->screen_mip_copy_view) {
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
    bridge->screen_mip_copy_view->setScene(bridge->scene);
    bridge->screen_mip_copy_view->setCamera(bridge->screen_mip_copy_camera);
    bridge->screen_mip_copy_view->setVisibleLayers(
            0xff, static_cast<uint8_t>(1u << kScreenMipCopyLayer));
    bridge->screen_mip_copy_view->setPostProcessingEnabled(false);
    bridge->screen_mip_copy_view->setAntiAliasing(filament::AntiAliasing::NONE);
    // The sampler is required by the material. Keep the renderable detached
    // until a valid runtime Vulkan image has been imported.
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
            if (!bridge->screen_in_scene && !bridge->screen_entity.isNull()) {
                bridge->scene->addEntity(bridge->screen_entity);
                bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, true);
                bridge->screen_in_scene = true;
            }
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
    if (!bridge->screen_in_scene && !bridge->screen_entity.isNull()) {
        bridge->scene->addEntity(bridge->screen_entity);
        bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, true);
        bridge->screen_in_scene = true;
    }
    return 1;
}

int bridge_screen_prepare_frame(FilamentBridge* bridge) {
    if (!bridge || !bridge->frame_active || !bridge->renderer ||
            !bridge->screen_mip_experiment_enabled ||
            bridge->active_eye >= bridge->screen_mip_textures.size() ||
            !bridge->screen_source_textures[bridge->active_eye] ||
            !bridge->screen_mip_copy_view ||
            !bridge->screen_mip_copy_material_instance) {
        return 0;
    }
    auto* source = bridge->screen_source_textures[bridge->active_eye];
    if (!ensure_screen_quality_targets(
            bridge, bridge->active_eye, source->getWidth(), source->getHeight(),
            bridge->screen_source_formats[bridge->active_eye])) {
        return 0;
    }
    auto* lanczos_texture = bridge->screen_lanczos_textures[bridge->active_eye];
    auto* mip_texture = bridge->screen_mip_textures[bridge->active_eye];
    auto* lanczos_target = bridge->screen_lanczos_render_targets[bridge->active_eye];
    auto* mip_target = bridge->screen_mip_render_targets[bridge->active_eye];

    // Pass 1: reconstruct the external source into the bounded quality target.
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexture", source, bridge->screen_source_texture_sampler);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexelSize", filament::math::float2{
                    1.0f / static_cast<float>(source->getWidth()),
                    1.0f / static_cast<float>(source->getHeight())});
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenQualityPass", 1.0f);
    bridge->screen_mip_copy_view->setRenderTarget(lanczos_target);
    bridge->screen_mip_copy_view->setViewport(filament::Viewport{
            0, 0, static_cast<uint32_t>(lanczos_texture->getWidth()),
            static_cast<uint32_t>(lanczos_texture->getHeight())});
    bridge->renderer->render(bridge->screen_mip_copy_view);

    // Pass 2: sharpen the reconstructed image, then generate the stable MIP
    // chain from the final quality result for video minification.
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexture", lanczos_texture,
            bridge->screen_source_texture_sampler);
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenTexelSize", filament::math::float2{
                    1.0f / static_cast<float>(lanczos_texture->getWidth()),
                    1.0f / static_cast<float>(lanczos_texture->getHeight())});
    bridge->screen_mip_copy_material_instance->setParameter(
            "screenQualityPass", 2.0f);
    bridge->screen_mip_copy_view->setRenderTarget(mip_target);
    bridge->screen_mip_copy_view->setViewport(filament::Viewport{
            0, 0, static_cast<uint32_t>(mip_texture->getWidth()),
            static_cast<uint32_t>(mip_texture->getHeight())});
    bridge->renderer->render(bridge->screen_mip_copy_view);
    mip_texture->generateMipmaps(*bridge->engine);
    ++bridge->screen_mip_generation_count[bridge->active_eye];
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
    if (!bridge->screen_in_scene && !bridge->screen_entity.isNull()) {
        bridge->scene->addEntity(bridge->screen_entity);
        bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, true);
        bridge->screen_in_scene = true;
    }
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
