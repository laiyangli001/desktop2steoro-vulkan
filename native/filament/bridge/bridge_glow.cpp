#include "bridge_glow.h"
#include "bridge_internal.h"

#include <backend/PixelBufferDescriptor.h>

#include <cstring>
#include <initializer_list>

namespace {

constexpr uint32_t kGlowSegments = 64;
constexpr uint32_t kGlowShellSegments = 96;
constexpr uint32_t kGlowShellRadialSegments = 48;
constexpr uint32_t kFlatFrostDepthSteps = 8;
constexpr uint32_t kFlatFrostEdgeSteps = 8;
constexpr uint32_t kMaxGlowVertices = (kGlowSegments + 1) * 2;
constexpr uint32_t kMaxGlowIndices = kGlowSegments * 6;
constexpr uint32_t kMaxGlowShellVertices =
        4 * (kGlowShellSegments + 1) * (kGlowShellRadialSegments + 1);
constexpr uint32_t kMaxGlowShellIndices =
        4 * kGlowShellSegments * kGlowShellRadialSegments * 6;
constexpr uint32_t kFlatFrostQuadCount =
        4 * kFlatFrostDepthSteps * kFlatFrostEdgeSteps;
constexpr uint32_t kMaxFrostVertices = kFlatFrostQuadCount * 4;
constexpr uint32_t kMaxFrostIndices = kFlatFrostQuadCount * 6;
constexpr float kCurvedHalfAngle = 0.72f;

uint8_t mip_level_count(uint32_t width, uint32_t height) {
    uint32_t maximum = std::max(width, height);
    uint8_t levels = 1;
    while (maximum > 1 && levels < 255) {
        maximum = (maximum + 1) / 2;
        ++levels;
    }
    return levels;
}

void append_quad(
        std::vector<GlowVertex>& vertices, std::vector<uint16_t>& indices,
        const GlowVertex& a, const GlowVertex& b,
        const GlowVertex& c, const GlowVertex& d) {
    const auto first = static_cast<uint16_t>(vertices.size());
    vertices.insert(vertices.end(), {a, b, c, d});
    indices.insert(indices.end(), {
            first, static_cast<uint16_t>(first + 1), static_cast<uint16_t>(first + 2),
            static_cast<uint16_t>(first + 1), static_cast<uint16_t>(first + 3),
            static_cast<uint16_t>(first + 2)});
}

filament::math::float3 screen_surface(
        const FilamentBridge* bridge, float u, float v) {
    const float half_width = bridge->screen_width * 0.5f;
    const float half_height = bridge->screen_height * 0.5f;
    float local_x = (u - 0.5f) * bridge->screen_width;
    float local_z = 0.0f;
    if (bridge->screen_curved) {
        const float radius = half_width / kCurvedHalfAngle;
        const float angle = -kCurvedHalfAngle + 2.0f * kCurvedHalfAngle * u;
        local_x = radius * std::sin(angle);
        local_z = radius * (1.0f - std::cos(angle));
    }
    return bridge->screen_center + bridge->screen_right * local_x +
            bridge->screen_up * ((v - 0.5f) * 2.0f * half_height) +
            bridge->screen_forward * local_z;
}

filament::math::float3 extended_screen_surface(
        const FilamentBridge* bridge, float local_x, float local_y) {
    const float half_width = std::max(bridge->screen_width * 0.5f, 1e-5f);
    const float half_height = std::max(bridge->screen_height * 0.5f, 1e-5f);
    const float clamped_x = std::clamp(local_x, -half_width, half_width);
    const float clamped_y = std::clamp(local_y, -half_height, half_height);
    const float u = clamped_x / bridge->screen_width + 0.5f;
    const float v = clamped_y / bridge->screen_height + 0.5f;
    auto result = screen_surface(bridge, u, v);
    auto tangent = bridge->screen_right;
    if (bridge->screen_curved) {
        const float angle = -kCurvedHalfAngle + 2.0f * kCurvedHalfAngle * u;
        tangent = bridge->screen_right * std::cos(angle) +
                bridge->screen_forward * std::sin(angle);
    }
    result += tangent * (local_x - clamped_x);
    result += bridge->screen_up * (local_y - clamped_y);
    return result;
}

float legacy_glow_range(const FilamentBridge* bridge) {
    const float base_width = std::max(bridge->glow_width, 0.75f);
    const float screen_long = std::max(
            std::max(bridge->screen_width, bridge->screen_height), 2.4f);
    const auto delta = bridge->glow_head_position - bridge->screen_center;
    const float distance = std::max(length(delta), 0.5f);
    return base_width * (screen_long / 2.4f) * (distance / 2.0f) * 20.0f;
}

void upload_geometry(
        FilamentBridge* bridge, filament::VertexBuffer* vertex_buffer,
        filament::IndexBuffer* index_buffer,
        const std::vector<GlowVertex>& vertices,
        const std::vector<uint16_t>& indices,
        std::initializer_list<utils::Entity> entities) {
    if (!bridge || !vertex_buffer || !index_buffer || vertices.empty() || indices.empty()) return;
    vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    vertices.data(), vertices.size() * sizeof(GlowVertex), nullptr));
    index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    indices.data(), indices.size() * sizeof(uint16_t), nullptr));
    auto& renderables = bridge->engine->getRenderableManager();
    for (const auto entity : entities) {
        const auto instance = renderables.getInstance(entity);
        if (!instance.isValid()) continue;
        renderables.setGeometryAt(
                instance, 0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                vertex_buffer, index_buffer, 0,
                static_cast<uint32_t>(indices.size()));
    }
}

void rebuild_glow_geometry(FilamentBridge* bridge) {
    if (!bridge || !bridge->glow_vertex_buffer || !bridge->glow_index_buffer) return;
    float range = std::max(legacy_glow_range(bridge), 1e-4f);
    if (bridge->glow_mode == 2) range *= 0.5f;
    const float glow_width = bridge->screen_width + range * 2.0f;
    const float glow_height = bridge->screen_height + range * 2.0f;
    bridge->glow_vertices.clear();
    bridge->glow_indices.clear();
    for (uint32_t segment = 0; segment <= kGlowSegments; ++segment) {
        const float u = static_cast<float>(segment) / static_cast<float>(kGlowSegments);
        const float local_x = (u - 0.5f) * glow_width;
        const auto bottom = extended_screen_surface(bridge, local_x, -glow_height * 0.5f);
        const auto top = extended_screen_surface(bridge, local_x, glow_height * 0.5f);
        bridge->glow_vertices.push_back({bottom, {u, 0.0f}, {0.0f, 0.0f}});
        bridge->glow_vertices.push_back({top, {u, 1.0f}, {0.0f, 0.0f}});
    }
    for (uint16_t segment = 0; segment < kGlowSegments; ++segment) {
        const uint16_t lower_left = segment * 2;
        const uint16_t upper_left = lower_left + 1;
        const uint16_t lower_right = lower_left + 2;
        const uint16_t upper_right = lower_left + 3;
        bridge->glow_indices.insert(bridge->glow_indices.end(), {
                lower_left, lower_right, upper_left,
                lower_right, upper_right, upper_left});
    }
    upload_geometry(
            bridge, bridge->glow_vertex_buffer, bridge->glow_index_buffer,
            bridge->glow_vertices, bridge->glow_indices,
            {bridge->glow_outer_entity, bridge->glow_inner_entity});

    const float inner_width = bridge->screen_width / glow_width;
    const float inner_height = bridge->screen_height / glow_height;
    const float inverse_range = std::max(glow_width, glow_height) / range;
    const float inner = bridge->glow_mode == 2 ? 0.0f :
            std::min(inner_width, inner_height) * 0.075f;
    for (auto* instance : {
            bridge->glow_outer_material_instance,
            bridge->glow_inner_material_instance}) {
        if (!instance) continue;
        instance->setParameter("screenHalf", filament::math::float2{
                inner_width * 0.5f, inner_height * 0.5f});
        instance->setParameter("glowInvRange", inverse_range);
        instance->setParameter("glowInner", inner);
        instance->setParameter("glowIntensity",
                bridge->glow_intensity * bridge->glow_intensity_multiplier);
    }
}

void rebuild_frost_geometry(FilamentBridge* bridge) {
    if (!bridge || !bridge->frost_vertex_buffer || !bridge->frost_index_buffer) return;
    bridge->frost_vertices.clear();
    bridge->frost_indices.clear();
    const float half_width = bridge->screen_width * 0.5f;
    const float half_height = bridge->screen_height * 0.5f;
    const auto local_head = bridge->glow_head_position - bridge->screen_center;
    const float head_x = dot(local_head, bridge->screen_right);
    const float head_y = dot(local_head, bridge->screen_up);
    const float head_z = dot(local_head, bridge->screen_forward);
    const float screen_distance = std::abs(head_z);
    const float front_depth = std::max(
            std::max(head_z + 0.55f, screen_distance + 0.35f), 0.75f);
    const float front_half_width = std::max(half_width, std::abs(head_x) + 0.65f);
    const float front_half_height = std::max(half_height, std::abs(head_y) + 0.65f);

    auto front = [&](float u, float v) {
        return bridge->screen_center +
                bridge->screen_right * ((u * 2.0f - 1.0f) * front_half_width) +
                bridge->screen_up * ((v * 2.0f - 1.0f) * front_half_height) +
                bridge->screen_forward * front_depth;
    };
    auto vertex = [](const filament::math::float3& position,
                     float u, float v, float depth) {
        return GlowVertex{position, {u, v}, {depth, 0.0f}};
    };
    if (!bridge->screen_curved) {
        // Match the v2.5 reference: four independent walls, each subdivided
        // along both depth and edge axes. This prevents perspective creases
        // and avoids cross-wall corner seams.
        auto wall = [&](float au, float av, float bu, float bv) {
            auto wall_point = [&](float edge_t, float depth_t) {
                const float u = au + edge_t * (bu - au);
                const float v = av + edge_t * (bv - av);
                const auto rear = screen_surface(bridge, u, v);
                const auto front_point = front(u, v);
                return vertex(rear + (front_point - rear) * depth_t,
                        u, v, depth_t);
            };
            for (uint32_t depth_step = 0;
                    depth_step < kFlatFrostDepthSteps; ++depth_step) {
                const float t0 = static_cast<float>(depth_step) /
                        static_cast<float>(kFlatFrostDepthSteps);
                const float t1 = static_cast<float>(depth_step + 1) /
                        static_cast<float>(kFlatFrostDepthSteps);
                for (uint32_t edge_step = 0;
                        edge_step < kFlatFrostEdgeSteps; ++edge_step) {
                    const float s0 = static_cast<float>(edge_step) /
                            static_cast<float>(kFlatFrostEdgeSteps);
                    const float s1 = static_cast<float>(edge_step + 1) /
                            static_cast<float>(kFlatFrostEdgeSteps);
                    append_quad(bridge->frost_vertices, bridge->frost_indices,
                            wall_point(s0, t0), wall_point(s1, t0),
                            wall_point(s0, t1), wall_point(s1, t1));
                }
            }
        };
        wall(0.0f, 1.0f, 1.0f, 1.0f);
        wall(0.0f, 0.0f, 1.0f, 0.0f);
        wall(0.0f, 1.0f, 0.0f, 0.0f);
        wall(1.0f, 1.0f, 1.0f, 0.0f);
    } else {
        for (uint32_t segment = 0; segment < kGlowSegments; ++segment) {
            const float u0 = static_cast<float>(segment) / kGlowSegments;
            const float u1 = static_cast<float>(segment + 1) / kGlowSegments;
            append_quad(bridge->frost_vertices, bridge->frost_indices,
                    vertex(screen_surface(bridge, u0, 0.0f), u0, 0.0f, 0.0f),
                    vertex(screen_surface(bridge, u1, 0.0f), u1, 0.0f, 0.0f),
                    vertex(front(u0, 0.0f), u0, 0.0f, 1.0f),
                    vertex(front(u1, 0.0f), u1, 0.0f, 1.0f));
            append_quad(bridge->frost_vertices, bridge->frost_indices,
                    vertex(screen_surface(bridge, u0, 1.0f), u0, 1.0f, 0.0f),
                    vertex(screen_surface(bridge, u1, 1.0f), u1, 1.0f, 0.0f),
                    vertex(front(u0, 1.0f), u0, 1.0f, 1.0f),
                    vertex(front(u1, 1.0f), u1, 1.0f, 1.0f));
        }
        for (float u : {0.0f, 1.0f}) {
            append_quad(bridge->frost_vertices, bridge->frost_indices,
                    vertex(screen_surface(bridge, u, 0.0f), u, 0.0f, 0.0f),
                    vertex(screen_surface(bridge, u, 1.0f), u, 1.0f, 0.0f),
                    vertex(front(u, 0.0f), u, 0.0f, 1.0f),
                    vertex(front(u, 1.0f), u, 1.0f, 1.0f));
        }
    }
    upload_geometry(
            bridge, bridge->frost_vertex_buffer, bridge->frost_index_buffer,
            bridge->frost_vertices, bridge->frost_indices,
            {bridge->frost_entity});
}

void rebuild_glow_shell_geometry(FilamentBridge* bridge) {
    if (!bridge || !bridge->glow_shell_vertex_buffer ||
            !bridge->glow_shell_index_buffer) return;
    const float radius = std::max(
            bridge->glow_shell_radius,
            std::max({bridge->screen_width, bridge->screen_height, 1.0f}) * 0.85f);
    const float height = std::max(
            bridge->glow_shell_height, bridge->screen_height * 1.8f);
    bridge->glow_shell_vertices.clear();
    bridge->glow_shell_indices.clear();
    auto shell_forward = bridge->screen_center - bridge->glow_head_position;
    if (length(shell_forward) <= 1e-5f) {
        shell_forward = -bridge->screen_forward;
    } else {
        shell_forward = normalize(shell_forward);
    }
    auto shell_right = bridge->screen_right -
            shell_forward * dot(bridge->screen_right, shell_forward);
    if (length(shell_right) <= 1e-5f) {
        shell_right = cross(shell_forward, bridge->screen_up);
    }
    shell_right = normalize(shell_right);
    const auto shell_up = normalize(cross(shell_right, shell_forward));
    const float vertical_radius = std::max(height * 0.5f, 1.0f);
    auto edge_uv = [](uint32_t side, float along) {
        switch (side) {
            case 0: return filament::math::float2{along, 1.0f};
            case 1: return filament::math::float2{1.0f, 1.0f - along};
            case 2: return filament::math::float2{1.0f - along, 0.0f};
            default: return filament::math::float2{0.0f, along};
        }
    };
    auto spherical_interpolate = [](
            const filament::math::float3& start,
            const filament::math::float3& end, float amount) {
        const float cosine = std::clamp(dot(start, end), -1.0f, 1.0f);
        const float angle = std::acos(cosine);
        const float sine = std::sin(angle);
        if (std::abs(sine) <= 1e-5f) {
            return normalize(start * (1.0f - amount) + end * amount);
        }
        return normalize(
                start * (std::sin((1.0f - amount) * angle) / sine) +
                end * (std::sin(amount * angle) / sine));
    };
    auto shell_position = [&](const filament::math::float3& direction) {
        // Intersect the original eye ray with the ellipsoid. Scaling each
        // basis component independently changes the ray direction whenever
        // the vertical and horizontal radii differ, which makes the apparent
        // emission boundary drift away from the live screen size.
        const float local_x = dot(direction, shell_right);
        const float local_y = dot(direction, shell_up);
        const float local_z = dot(direction, shell_forward);
        const float inverse_distance_squared =
                local_x * local_x / (radius * radius) +
                local_y * local_y / (vertical_radius * vertical_radius) +
                local_z * local_z / (radius * radius);
        const float intersection_distance = 1.0f /
                std::sqrt(std::max(inverse_distance_squared, 1e-8f));
        return bridge->glow_head_position +
                direction * intersection_distance;
    };

    // Four independent edge-to-rim strips replace the latitude/longitude
    // hemisphere. Every screen-edge sample owns its radial geodesic, so no
    // row of vertices can collapse into a shared top, bottom, left, or right
    // pole and make the surround light converge there.
    const uint32_t radial_stride = kGlowShellSegments + 1;
    for (uint32_t side = 0; side < 4; ++side) {
        const uint32_t side_first =
                static_cast<uint32_t>(bridge->glow_shell_vertices.size());
        for (uint32_t radial = 0; radial <= kGlowShellRadialSegments; ++radial) {
            const float radial_t = static_cast<float>(radial) /
                    static_cast<float>(kGlowShellRadialSegments);
            for (uint32_t segment = 0; segment <= kGlowShellSegments; ++segment) {
                const float along = static_cast<float>(segment) /
                        static_cast<float>(kGlowShellSegments);
                const auto source_uv = edge_uv(side, along);
                const auto source_position = screen_surface(
                        bridge, source_uv.x, source_uv.y);
                auto source_direction =
                        source_position - bridge->glow_head_position;
                float source_distance = length(source_direction);
                if (length(source_direction) <= 1e-5f) {
                    source_direction = shell_forward;
                    source_distance = 0.0f;
                } else {
                    source_direction = normalize(source_direction);
                }
                const float rim_x = (source_uv.x - 0.5f) * 2.0f;
                const float rim_y = (source_uv.y - 0.5f) * 2.0f;
                auto rim_direction = shell_right * rim_x + shell_up * rim_y +
                        shell_forward * 0.02f;
                rim_direction = normalize(rim_direction);
                const auto direction = spherical_interpolate(
                        source_direction, rim_direction, radial_t);
                const auto shell_target = shell_position(direction);
                const float shell_distance = length(
                        shell_target - bridge->glow_head_position);
                const float surface_distance =
                        source_distance * (1.0f - radial_t) +
                        shell_distance * radial_t;
                const auto surface_position = radial == 0
                        ? source_position
                        : bridge->glow_head_position +
                                direction * surface_distance;
                bridge->glow_shell_vertices.push_back({
                        surface_position, source_uv, {radial_t, 0.0f}});
            }
        }
        for (uint32_t radial = 0; radial < kGlowShellRadialSegments; ++radial) {
            for (uint32_t segment = 0; segment < kGlowShellSegments; ++segment) {
                const auto near_left = static_cast<uint16_t>(
                        side_first + radial * radial_stride + segment);
                const auto near_right = static_cast<uint16_t>(near_left + 1);
                const auto far_left = static_cast<uint16_t>(
                        near_left + radial_stride);
                const auto far_right = static_cast<uint16_t>(far_left + 1);
                bridge->glow_shell_indices.insert(
                        bridge->glow_shell_indices.end(), {
                                near_left, near_right, far_left,
                                near_right, far_right, far_left});
            }
        }
    }
    upload_geometry(
            bridge, bridge->glow_shell_vertex_buffer,
            bridge->glow_shell_index_buffer, bridge->glow_shell_vertices,
            bridge->glow_shell_indices, {bridge->glow_shell_entity});
}

filament::Material* build_material(
        FilamentBridge* bridge, const char* name, const char* shader,
        bool glow_material) {
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name(name)
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .parameter("glowTexture", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("glowIntensity", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("externalSource", filamat::MaterialBuilder::UniformType::FLOAT)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::TRANSPARENT)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false).depthCulling(false)
            .stereoscopicType(filamat::MaterialBuilder::StereoscopicType::MULTIVIEW)
            .stereoscopicEyeCount(2)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    if (glow_material) {
        builder.parameter("screenHalf", filamat::MaterialBuilder::UniformType::FLOAT2)
                .parameter("glowInvRange", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("glowInner", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("innerOnly", filamat::MaterialBuilder::UniformType::FLOAT);
    } else {
        builder.require(filament::VertexAttribute::UV1)
                .parameter("effectMode", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectAlpha", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectThreshold", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectLod", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectBlend", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectThickness", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectDiffuse", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectInset", filamat::MaterialBuilder::UniformType::FLOAT)
                .parameter("effectTime", filamat::MaterialBuilder::UniformType::FLOAT);
    }
    const auto package = builder.build(bridge->engine->getJobSystem());
    if (!package.isValid()) return nullptr;
    return filament::Material::Builder()
            .package(package.getData(), package.getSize()).build(*bridge->engine);
}

filament::Material* build_glow_shell_material(
        FilamentBridge* bridge, const char* shader) {
    filamat::MaterialBuilder::init();
    const auto package = filamat::MaterialBuilder()
            .name("D2S Legacy Surround Glow")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .require(filament::VertexAttribute::UV1)
            .parameter("glowTexture", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .parameter("glowIntensity", filamat::MaterialBuilder::UniformType::FLOAT)
            .parameter("externalSource", filamat::MaterialBuilder::UniformType::FLOAT)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::ADD)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false).depthCulling(false)
            .stereoscopicType(filamat::MaterialBuilder::StereoscopicType::MULTIVIEW)
            .stereoscopicEyeCount(2)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL)
            .build(bridge->engine->getJobSystem());
    if (!package.isValid()) return nullptr;
    return filament::Material::Builder()
            .package(package.getData(), package.getSize()).build(*bridge->engine);
}

bool create_renderable(
        FilamentBridge* bridge, utils::Entity& entity,
        filament::MaterialInstance* material,
        filament::VertexBuffer* vertices, filament::IndexBuffer* indices,
        uint32_t index_count, uint8_t priority) {
    entity = utils::EntityManager::get().create();
    const auto result = filament::RenderableManager::Builder(1)
            .boundingBox({{-20000.0f, -20000.0f, -20000.0f},
                          {20000.0f, 20000.0f, 20000.0f}})
            .material(0, material)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    vertices, indices, 0, index_count)
            .priority(priority).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, entity);
    if (result != filament::RenderableManager::Builder::Success) return false;
    bridge->foreground_scene->addEntity(entity);
    bridge_set_renderable_layer(bridge, entity, 1, false);
    return true;
}

void bind_source(FilamentBridge* bridge) {
    if (!bridge || !bridge->glow_source_texture) return;
    const auto& sampler = bridge->glow_source_external
            ? bridge->glow_external_texture_sampler
            : bridge->glow_texture_sampler;
    for (auto* instance : {
            bridge->glow_outer_material_instance,
            bridge->glow_inner_material_instance,
            bridge->frost_material_instance,
            bridge->veil_material_instance,
            bridge->glow_shell_material_instance}) {
        if (!instance) continue;
        instance->setParameter("glowTexture", bridge->glow_source_texture, sampler);
        instance->setParameter(
                "externalSource", bridge->glow_source_external ? 1.0f : 0.0f);
    }
}

}  // namespace

void bridge_glow_destroy(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    for (auto entity : {
            bridge->glow_outer_entity, bridge->glow_inner_entity,
            bridge->frost_entity}) {
        if (entity.isNull()) continue;
        if (bridge->foreground_scene) bridge->foreground_scene->remove(entity);
        bridge->engine->destroy(entity);
    }
    if (!bridge->glow_shell_entity.isNull()) {
        // The shell belongs to the room/background pass. Remove from both
        // scenes so partial create failures remain safe to destroy.
        if (bridge->scene) bridge->scene->remove(bridge->glow_shell_entity);
        if (bridge->foreground_scene) {
            bridge->foreground_scene->remove(bridge->glow_shell_entity);
        }
        bridge->engine->destroy(bridge->glow_shell_entity);
    }
    bridge->glow_outer_entity = {};
    bridge->glow_inner_entity = {};
    bridge->frost_entity = {};
    bridge->glow_shell_entity = {};
    for (auto** buffer : {
            &bridge->glow_vertex_buffer, &bridge->frost_vertex_buffer,
            &bridge->glow_shell_vertex_buffer}) {
        if (*buffer) bridge->engine->destroy(*buffer);
        *buffer = nullptr;
    }
    for (auto** buffer : {
            &bridge->glow_index_buffer, &bridge->frost_index_buffer,
            &bridge->glow_shell_index_buffer}) {
        if (*buffer) bridge->engine->destroy(*buffer);
        *buffer = nullptr;
    }
    for (auto** instance : {
            &bridge->glow_outer_material_instance, &bridge->glow_inner_material_instance,
            &bridge->frost_material_instance, &bridge->veil_material_instance,
            &bridge->glow_shell_material_instance}) {
        if (*instance) bridge->engine->destroy(*instance);
        *instance = nullptr;
    }
    for (auto** material : {
            &bridge->glow_material, &bridge->frost_material,
            &bridge->glow_shell_material}) {
        if (*material) bridge->engine->destroy(*material);
        *material = nullptr;
    }
    if (bridge->glow_cpu_source_texture) {
        bridge->engine->destroy(bridge->glow_cpu_source_texture);
        bridge->glow_cpu_source_texture = nullptr;
    }
    for (auto& slot : bridge->glow_texture_cache) {
        if (slot.texture) {
            bridge->engine->destroy(slot.texture);
            slot.texture = nullptr;
        }
    }
    bridge->glow_texture_cache.clear();
    bridge->glow_source_texture = nullptr;
    bridge->glow_source_external = false;
    bridge->glow_vertices.clear();
    bridge->glow_indices.clear();
    bridge->frost_vertices.clear();
    bridge->frost_indices.clear();
    bridge->glow_shell_vertices.clear();
    bridge->glow_shell_indices.clear();
}

int bridge_glow_create(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine || !bridge->scene ||
            !bridge->foreground_scene) return 0;
    bridge_glow_destroy(bridge);
    bridge->glow_texture_sampler = filament::TextureSampler(
            filament::TextureSampler::MinFilter::LINEAR_MIPMAP_LINEAR,
            filament::TextureSampler::MagFilter::LINEAR,
            filament::TextureSampler::WrapMode::CLAMP_TO_EDGE);
    bridge->glow_texture_sampler.setAnisotropy(16.0f);
    bridge->glow_external_texture_sampler = filament::TextureSampler(
            filament::TextureSampler::MinFilter::LINEAR,
            filament::TextureSampler::MagFilter::LINEAR,
            filament::TextureSampler::WrapMode::CLAMP_TO_EDGE);
    bridge->glow_external_texture_sampler.setAnisotropy(16.0f);
    const char* glow_shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            vec2 uv = getUV0();
            vec2 centered = uv - vec2(0.5);
            vec2 p = abs(centered) - materialParams.screenHalf;
            float signedDistance = max(p.x, p.y);
            float inner = max(materialParams.glowInner, 0.0);
            bool inside = materialParams.innerOnly > 0.5;
            if (inside) {
                if (inner <= 0.0 || signedDistance > 0.0 || signedDistance < -inner) discard;
            } else if (signedDistance <= 0.0) {
                discard;
            }
            float edgePosition = inside ? signedDistance + inner : signedDistance;
            float edgeT = clamp(edgePosition * max(materialParams.glowInvRange, 0.001), 0.0, 1.0);
            float glow = pow(max(1.0 - smoothstep(0.0, 1.0, edgeT), 0.0), 0.58);
            if (inside) {
                float density = smoothstep(0.0, max(inner, 0.000001), signedDistance + inner);
                density = density * density * (3.0 - 2.0 * density);
                glow *= density;
            }
            glow *= materialParams.glowIntensity;
            if (glow <= 0.001) discard;
            vec2 raw = (uv - (vec2(0.5) - materialParams.screenHalf)) /
                    max(materialParams.screenHalf * 2.0, vec2(0.00001));
            vec2 contentUv = vec2(raw.x, 1.0 - raw.y);
            vec2 direction = normalize(contentUv - vec2(0.5) + vec2(0.00001, 0.0));
            vec2 denominator = max(abs(direction), vec2(0.00001));
            float edgeScale = min(0.5 / denominator.x, 0.5 / denominator.y);
            vec2 edgeUv = clamp(vec2(0.5) + direction * edgeScale, vec2(0.0), vec2(1.0));
            float blur = smoothstep(0.0, 1.0, edgeT);
            vec2 lightUv = edgeUv - direction * mix(0.015, 0.060, blur);
            float lod = materialParams.externalSource > 0.5
                    ? 0.0 : mix(7.0, 9.0, blur);
            vec3 localColor = textureLod(materialParams_glowTexture,
                    clamp(lightUv, vec2(0.0), vec2(1.0)), lod).rgb;
            vec3 edgeColor = textureLod(materialParams_glowTexture, edgeUv, lod + 1.0).rgb;
            vec3 color = mix(localColor, edgeColor, blur * 0.5);
            glow = min(glow, 1.0);
            // Transparency is fixed at zero: every surviving Glow fragment is
            // fully opaque while RGB still carries the authored falloff.
            material.baseColor = vec4(color * glow, 1.0);
        }
    )FILAMENT";
    const char* frost_shader = R"FILAMENT(
        float glowHash(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            vec2 uv = clamp(getUV0(), vec2(0.0), vec2(1.0));
            float depth = clamp(getUV1().x, 0.0, 1.0);
            float beam = exp(-depth / 0.34);
            beam = pow(max(beam, 0.0), 1.0 / max(materialParams.effectThickness, 0.1));
            float edgeDistance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            float inset = max(materialParams.effectInset, 0.0001);
            float edge = 1.0 - smoothstep(inset, inset * 4.0, edgeDistance);
            vec2 sampleUv = vec2(uv.x, 1.0 - uv.y);
            float sourceLod = materialParams.externalSource > 0.5
                    ? 0.0 : max(materialParams.effectLod, 0.0);
            vec3 source = textureLod(
                    materialParams_glowTexture, sampleUv, sourceLod).rgb;
            float alpha;
            vec3 color = source;
            if (materialParams.effectMode < 0.5) {
                alpha = edge * beam * materialParams.effectAlpha * materialParams.glowIntensity;
            } else {
                float noise = glowHash(floor((sampleUv + vec2(materialParams.effectTime * 0.011,
                        -materialParams.effectTime * 0.007)) * 54.0));
                float luma = dot(source, vec3(0.2126, 0.7152, 0.0722));
                float bright = smoothstep(materialParams.effectThreshold, 1.0, luma);
                float scatter = max(bright, luma * clamp(materialParams.effectDiffuse, 0.0, 2.0) * 0.35);
                alpha = edge * beam * scatter * materialParams.effectAlpha *
                        materialParams.glowIntensity * (0.82 + 0.30 * noise);
                color = mix(source, vec3(luma), 0.28);
                color = color * (0.55 + materialParams.effectBlend * 0.35) +
                        source * bright * 0.35;
            }
            if (alpha <= 0.002) discard;
            alpha = min(alpha, 1.0);
            // Veil and Frosted use alpha only as an effect-strength mask. The
            // emitted fragment itself is fully opaque.
            material.baseColor = vec4(color * alpha, 1.0);
        }
    )FILAMENT";
    const char* glow_shell_shader = R"FILAMENT(
        vec3 sampleRegionCell(vec2 cell, vec2 grid) {
            vec2 q = (clamp(cell, vec2(0.0), grid - vec2(1.0)) +
                    vec2(0.5)) / grid;
            q.y = 1.0 - q.y;
            return textureLod(materialParams_glowTexture, q, 0.0).rgb;
        }
        vec3 sampleRegionAverage(vec2 p) {
            vec2 grid = vec2(8.0, 6.0);
            vec2 position = clamp(p, vec2(0.0), vec2(1.0)) * grid - vec2(0.5);
            vec2 base = floor(position);
            vec2 blend = fract(position);
            blend = blend * blend * (vec2(3.0) - vec2(2.0) * blend);
            vec3 lower = mix(
                    sampleRegionCell(base, grid),
                    sampleRegionCell(base + vec2(1.0, 0.0), grid), blend.x);
            vec3 upper = mix(
                    sampleRegionCell(base + vec2(0.0, 1.0), grid),
                    sampleRegionCell(base + vec2(1.0, 1.0), grid), blend.x);
            return mix(lower, upper, blend.y);
        }
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            // UV0 stays on one screen edge while UV1 advances along that
            // sample's independent geodesic to the hemisphere rim.
            vec2 sourceUv = getUV0();
            float radialDistance = clamp(getUV1().x, 0.0, 1.0);
            // Start at full edge intensity and follow the long exponential
            // tail. Seam color matching is handled by the producer's narrow
            // screen-edge pixel bands, without dimming the first glow row.
            float edgeField = exp2(-5.0 * radialDistance) *
                    (1.0 - smoothstep(0.88, 1.0, radialDistance));
            float glow = edgeField * materialParams.glowIntensity;
            if (glow <= 0.002) discard;
            glow = min(glow, 1.0);
            // Clamp to the nearest point on the screen perimeter. The 8 x 6
            // producer grid averages a narrow inward pixel band for each edge
            // segment, then this smooth interpolation removes cell boundaries.
            vec3 shellColor = sampleRegionAverage(sourceUv);
            // Surround is emitted light, not a translucent colored wall.
            // Additive output lets black samples contribute zero instead of
            // producing an opaque black rectangle behind the screen.
            material.baseColor = vec4(shellColor * glow, 1.0);
        }
    )FILAMENT";
    bridge->glow_material = build_material(
            bridge, "D2S Legacy Screen Glow", glow_shader, true);
    bridge->frost_material = build_material(
            bridge, "D2S Legacy Frosted Glow", frost_shader, false);
    bridge->glow_shell_material = build_glow_shell_material(
            bridge, glow_shell_shader);
    if (!bridge->glow_material || !bridge->frost_material ||
            !bridge->glow_shell_material) {
        bridge_set_error(bridge, "Filament could not build legacy glow materials");
        return 0;
    }
    bridge->glow_outer_material_instance = bridge->glow_material->createInstance();
    bridge->glow_inner_material_instance = bridge->glow_material->createInstance();
    bridge->frost_material_instance = bridge->frost_material->createInstance();
    bridge->veil_material_instance = bridge->frost_material->createInstance();
    bridge->glow_shell_material_instance =
            bridge->glow_shell_material->createInstance();
    bridge->glow_outer_material_instance->setParameter("innerOnly", 0.0f);
    bridge->glow_inner_material_instance->setParameter("innerOnly", 1.0f);
    bridge->glow_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(kMaxGlowVertices).bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3, 0, sizeof(GlowVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(GlowVertex)).build(*bridge->engine);
    bridge->glow_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(kMaxGlowIndices)
            .bufferType(filament::IndexBuffer::IndexType::USHORT).build(*bridge->engine);
    bridge->frost_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(kMaxFrostVertices).bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3, 0, sizeof(GlowVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(GlowVertex))
            .attribute(filament::VertexAttribute::UV1, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 5, sizeof(GlowVertex)).build(*bridge->engine);
    bridge->frost_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(kMaxFrostIndices)
            .bufferType(filament::IndexBuffer::IndexType::USHORT).build(*bridge->engine);
    bridge->glow_shell_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(kMaxGlowShellVertices).bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3, 0, sizeof(GlowVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(GlowVertex))
            .attribute(filament::VertexAttribute::UV1, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 5, sizeof(GlowVertex)).build(*bridge->engine);
    bridge->glow_shell_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(kMaxGlowShellIndices)
            .bufferType(filament::IndexBuffer::IndexType::USHORT).build(*bridge->engine);
    if (!bridge->glow_outer_material_instance || !bridge->glow_inner_material_instance ||
            !bridge->frost_material_instance || !bridge->veil_material_instance ||
            !bridge->glow_shell_material_instance ||
            !bridge->glow_vertex_buffer || !bridge->glow_index_buffer ||
            !bridge->frost_vertex_buffer || !bridge->frost_index_buffer ||
            !bridge->glow_shell_vertex_buffer || !bridge->glow_shell_index_buffer ||
            !create_renderable(bridge, bridge->glow_outer_entity,
                    bridge->glow_outer_material_instance, bridge->glow_vertex_buffer,
                    bridge->glow_index_buffer, kMaxGlowIndices, 4) ||
            !create_renderable(bridge, bridge->glow_inner_entity,
                    bridge->glow_inner_material_instance, bridge->glow_vertex_buffer,
                    bridge->glow_index_buffer, kMaxGlowIndices, 5) ||
            !create_renderable(bridge, bridge->frost_entity,
                    bridge->frost_material_instance, bridge->frost_vertex_buffer,
                    bridge->frost_index_buffer, kMaxFrostIndices, 5) ||
            !create_renderable(bridge, bridge->glow_shell_entity,
                    bridge->glow_shell_material_instance,
                    bridge->glow_shell_vertex_buffer,
                    bridge->glow_shell_index_buffer,
                    kMaxGlowShellIndices, 0)) {
        bridge_set_error(bridge, "Filament could not create legacy glow geometry");
        return 0;
    }
    // Legacy surround is a screen-background effect. Put only this shell in
    // the room Scene so the main View finishes before the foreground View
    // draws the opaque virtual screen and edge effects.
    bridge->foreground_scene->remove(bridge->glow_shell_entity);
    bridge->scene->addEntity(bridge->glow_shell_entity);
    const uint8_t fallback[] = {77, 153, 255, 255};
    if (!bridge_glow_set_source(bridge, fallback, 1, 1)) return 0;
    bridge_glow_update_geometry(bridge);
    return 1;
}

int bridge_glow_set_source(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height) {
    if (!bridge || !bridge->engine || !rgba || width == 0 || height == 0) return 0;
    if (!bridge->glow_cpu_source_texture ||
            bridge->glow_cpu_source_texture->getWidth() != width ||
            bridge->glow_cpu_source_texture->getHeight() != height) {
        if (bridge->glow_cpu_source_texture) {
            bridge->engine->destroy(bridge->glow_cpu_source_texture);
        }
        using Usage = filament::Texture::Usage;
        const auto glow_texture_usage = static_cast<Usage>(
                static_cast<uint16_t>(Usage::DEFAULT) |
                static_cast<uint16_t>(Usage::GEN_MIPMAPPABLE));
        bridge->glow_cpu_source_texture = filament::Texture::Builder()
                .width(width).height(height).levels(mip_level_count(width, height))
                .format(filament::Texture::InternalFormat::SRGB8_A8)
                .sampler(filament::Texture::Sampler::SAMPLER_2D)
                .usage(glow_texture_usage)
                .build(*bridge->engine);
        if (!bridge->glow_cpu_source_texture) {
            bridge_set_error(bridge, "Filament could not create CPU glow source texture");
            return 0;
        }
    }
    bridge->glow_source_texture = bridge->glow_cpu_source_texture;
    bridge->glow_source_external = false;
    bind_source(bridge);
    const size_t byte_count = static_cast<size_t>(width) * height * 4u;
    auto* pixels = new uint8_t[byte_count];
    std::memcpy(pixels, rgba, byte_count);
    filament::Texture::PixelBufferDescriptor descriptor(
            pixels, byte_count, filament::Texture::Format::RGBA,
            filament::Texture::Type::UBYTE,
            [](void* buffer, size_t, void*) { delete[] static_cast<uint8_t*>(buffer); });
    bridge->glow_cpu_source_texture->setImage(*bridge->engine, 0, std::move(descriptor));
    bridge->glow_cpu_source_texture->generateMipmaps(*bridge->engine);
    return 1;
}

int bridge_glow_set_image(
        FilamentBridge* bridge, const void* image,
        uint32_t width, uint32_t height, int32_t format) {
    if (!bridge || !bridge->engine || !bridge->platform ||
            !image || width == 0 || height == 0) return 0;
    if (format != VK_FORMAT_R8G8B8A8_SRGB &&
            format != VK_FORMAT_R8G8B8A8_UNORM) {
        bridge_set_error(bridge,
                "Glow source requires VK_FORMAT_R8G8B8A8_SRGB or VK_FORMAT_R8G8B8A8_UNORM");
        return 0;
    }
    for (const auto& slot : bridge->glow_texture_cache) {
        if (slot.image == image && slot.width == width &&
                slot.height == height && slot.format == format && slot.texture) {
            bridge->glow_source_texture = slot.texture;
            bridge->glow_source_external = true;
            bind_source(bridge);
            return 1;
        }
    }
#if defined(D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE)
    const auto external_image = bridge->platform->createExternalImageFromVkImage(
            reinterpret_cast<VkImage>(const_cast<void*>(image)),
            static_cast<VkFormat>(format), width, height);
    if (!external_image) {
        bridge_set_error(bridge,
                "Filament Vulkan backend rejected the Glow external VkImage metadata");
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
    if (texture) texture->setExternalImage(*bridge->engine, external_image);
    if (!texture) {
        bridge_set_error(bridge, "Filament could not import Glow Vulkan image");
        return 0;
    }
    bridge->glow_texture_cache.push_back(
            ScreenTextureSlot{image, texture, width, height, format});
    bridge->glow_source_texture = texture;
    bridge->glow_source_external = true;
    bind_source(bridge);
    return 1;
#else
    bridge_set_error(bridge,
            "Filament Glow external-image support is unavailable in this build");
    return 0;
#endif
}

int bridge_glow_set_state(
        FilamentBridge* bridge, uint32_t mode,
        float head_x, float head_y, float head_z,
        float glow_intensity, float glow_width,
        float glow_intensity_multiplier,
        float frosted_intensity, float frosted_alpha,
        float frosted_threshold, float frosted_lod,
        float frosted_blend, float frosted_thickness,
        float frosted_diffuse, float frosted_inset,
        float veil_intensity, float veil_alpha,
        float glow_shell_intensity_multiplier,
        float glow_shell_radius, float glow_shell_height) {
    if (!bridge || !bridge->engine || mode > 5) return 0;
    const float values[] = {
            head_x, head_y, head_z, glow_intensity, glow_width,
            glow_intensity_multiplier, frosted_intensity, frosted_alpha,
            frosted_threshold, frosted_lod, frosted_blend, frosted_thickness,
            frosted_diffuse, frosted_inset, veil_intensity, veil_alpha,
            glow_shell_intensity_multiplier, glow_shell_radius,
            glow_shell_height};
    for (float value : values) if (!std::isfinite(value)) return 0;
    const filament::math::float3 requested_head{head_x, head_y, head_z};
    const bool geometry_changed = mode != bridge->glow_mode ||
            std::abs(glow_width - bridge->glow_width) > 1e-4f ||
            std::abs(glow_shell_radius - bridge->glow_shell_radius) > 1e-4f ||
            std::abs(glow_shell_height - bridge->glow_shell_height) > 1e-4f ||
            length(requested_head - bridge->glow_head_position) >= 0.02f;
    bridge->glow_mode = mode;
    if (geometry_changed) bridge->glow_head_position = requested_head;
    bridge->glow_intensity = std::max(glow_intensity, 0.0f);
    bridge->glow_width = std::max(glow_width, 0.0f);
    bridge->glow_intensity_multiplier = std::max(glow_intensity_multiplier, 0.0f);
    bridge->frosted_intensity = std::max(frosted_intensity, 0.0f);
    bridge->frosted_alpha = std::clamp(frosted_alpha, 0.0f, 1.0f);
    bridge->frosted_threshold = std::clamp(frosted_threshold, 0.0f, 1.0f);
    bridge->frosted_lod = std::max(frosted_lod, 0.0f);
    bridge->frosted_blend = std::max(frosted_blend, 0.0f);
    bridge->frosted_thickness = std::max(frosted_thickness, 0.1f);
    bridge->frosted_diffuse = std::max(frosted_diffuse, 0.0f);
    bridge->frosted_inset = std::max(frosted_inset, 0.0001f);
    bridge->veil_intensity = std::max(veil_intensity, 0.0f);
    bridge->veil_alpha = std::clamp(veil_alpha, 0.0f, 1.0f);
    bridge->glow_shell_intensity_multiplier = std::max(
            glow_shell_intensity_multiplier, 0.0f);
    bridge->glow_shell_radius = std::max(glow_shell_radius, 0.0f);
    bridge->glow_shell_height = std::max(glow_shell_height, 0.0f);
    for (auto* instance : {
            bridge->glow_outer_material_instance,
            bridge->glow_inner_material_instance}) {
        if (instance) instance->setParameter(
                "glowIntensity",
                bridge->glow_intensity * bridge->glow_intensity_multiplier);
    }
    const float now = static_cast<float>(std::fmod(
            std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count(), 1024.0));
    if (bridge->frost_material_instance) {
        bridge->frost_material_instance->setParameter("effectMode", 1.0f);
        bridge->frost_material_instance->setParameter("glowIntensity",
                bridge->frosted_intensity * bridge->glow_intensity_multiplier);
        bridge->frost_material_instance->setParameter("effectAlpha", bridge->frosted_alpha);
        bridge->frost_material_instance->setParameter("effectThreshold", bridge->frosted_threshold);
        bridge->frost_material_instance->setParameter("effectLod", bridge->frosted_lod);
        bridge->frost_material_instance->setParameter("effectBlend", bridge->frosted_blend);
        bridge->frost_material_instance->setParameter("effectThickness", bridge->frosted_thickness);
        bridge->frost_material_instance->setParameter("effectDiffuse", bridge->frosted_diffuse);
        bridge->frost_material_instance->setParameter("effectInset", bridge->frosted_inset);
        bridge->frost_material_instance->setParameter("effectTime", now);
    }
    if (bridge->veil_material_instance) {
        bridge->veil_material_instance->setParameter("effectMode", 0.0f);
        bridge->veil_material_instance->setParameter("glowIntensity",
                bridge->veil_intensity * bridge->glow_intensity_multiplier);
        bridge->veil_material_instance->setParameter("effectAlpha", bridge->veil_alpha);
        bridge->veil_material_instance->setParameter("effectThreshold", 0.0f);
        bridge->veil_material_instance->setParameter("effectLod", 0.0f);
        bridge->veil_material_instance->setParameter("effectBlend", 1.0f);
        bridge->veil_material_instance->setParameter("effectThickness", 3.0f);
        bridge->veil_material_instance->setParameter("effectDiffuse", 0.0f);
        bridge->veil_material_instance->setParameter("effectInset", 0.02f);
        bridge->veil_material_instance->setParameter("effectTime", now);
    }
    if (bridge->glow_shell_material_instance) {
        bridge->glow_shell_material_instance->setParameter(
                "glowIntensity", bridge->glow_intensity *
                        bridge->glow_shell_intensity_multiplier);
    }
    if (!bridge->frost_entity.isNull()) {
        auto& renderables = bridge->engine->getRenderableManager();
        const auto instance = renderables.getInstance(bridge->frost_entity);
        if (instance.isValid()) renderables.setMaterialInstanceAt(
                instance, 0, mode == 3 ? bridge->veil_material_instance
                                      : bridge->frost_material_instance);
    }
    if (geometry_changed) bridge_glow_update_geometry(bridge);
    bridge_glow_update_visibility(bridge);
    return 1;
}

void bridge_glow_update_geometry(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    rebuild_glow_geometry(bridge);
    rebuild_frost_geometry(bridge);
    rebuild_glow_shell_geometry(bridge);
}

void bridge_glow_update_visibility(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    const bool enabled = bridge->glow_intensity_multiplier > 0.0f &&
            !bridge->passthrough_backdrop;
    const bool shell_enabled = bridge->glow_shell_intensity_multiplier > 0.0f &&
            !bridge->passthrough_backdrop;
    const bool default_environment = bridge->asset == nullptr;
    bridge_set_renderable_layer(bridge, bridge->glow_outer_entity, 1,
            enabled && default_environment &&
                    (bridge->glow_mode == 1 || bridge->glow_mode == 2));
    bridge_set_renderable_layer(bridge, bridge->glow_inner_entity, 1,
            enabled && default_environment && bridge->glow_mode == 1);
    bridge_set_renderable_layer(bridge, bridge->frost_entity, 1,
            enabled && default_environment &&
                    (bridge->glow_mode == 3 || bridge->glow_mode == 4));
    bridge_set_renderable_layer(bridge, bridge->glow_shell_entity, 1,
            shell_enabled && default_environment && bridge->glow_mode == 5);
}
