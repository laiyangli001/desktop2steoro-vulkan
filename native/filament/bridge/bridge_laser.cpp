#include "bridge_laser.h"
#include "bridge_internal.h"

void bridge_laser_destroy(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    for (auto& entity : bridge->laser_entities) {
        if (entity.isNull()) continue;
        if (bridge->scene) bridge->scene->remove(entity);
        bridge->engine->destroy(entity);
        entity = {};
    }
    if (bridge->laser_vertex_buffer) {
        bridge->engine->destroy(bridge->laser_vertex_buffer);
        bridge->laser_vertex_buffer = nullptr;
    }
    if (bridge->laser_index_buffer) {
        bridge->engine->destroy(bridge->laser_index_buffer);
        bridge->laser_index_buffer = nullptr;
    }
    if (bridge->laser_material_instance) {
        bridge->engine->destroy(bridge->laser_material_instance);
        bridge->laser_material_instance = nullptr;
    }
    if (bridge->laser_material) {
        bridge->engine->destroy(bridge->laser_material);
        bridge->laser_material = nullptr;
    }
}

int bridge_laser_create(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine || !bridge->scene) return 0;
    bridge_laser_destroy(bridge);
    const char* shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            float2 uv = getUV0();
            float edge = smoothstep(0.0, 0.18, uv.x) *
                    smoothstep(0.0, 0.18, 1.0 - uv.x);
            float fade = 1.0 - smoothstep(0.72, 1.0, uv.y);
            material.baseColor = float4(0.08, 0.62, 1.0, edge * fade * 0.9);
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S Controller Laser")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::TRANSPARENT)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false)
            .depthCulling(true)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    const filamat::Package package = builder.build(bridge->engine->getJobSystem());
    if (!package.isValid()) {
        bridge_set_error(bridge, "Filament could not build controller laser material");
        return 0;
    }
    bridge->laser_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*bridge->engine);
    if (!bridge->laser_material) {
        bridge_set_error(bridge, "Filament could not create controller laser material");
        return 0;
    }
    bridge->laser_material_instance = bridge->laser_material->createInstance();
    bridge->laser_vertices = {{
            {{-0.5f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{ 0.5f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{-0.5f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 0.5f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    }};
    bridge->laser_indices = {{0, 1, 2, 1, 3, 2}};
    bridge->laser_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(static_cast<uint32_t>(bridge->laser_vertices.size()))
            .bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(PreviewScreenVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(PreviewScreenVertex))
            .build(*bridge->engine);
    bridge->laser_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(static_cast<uint32_t>(bridge->laser_indices.size()))
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*bridge->engine);
    if (!bridge->laser_material_instance || !bridge->laser_vertex_buffer ||
            !bridge->laser_index_buffer) {
        bridge_set_error(bridge, "Filament could not create controller laser geometry");
        return 0;
    }
    bridge->laser_vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    bridge->laser_vertices.data(),
                    bridge->laser_vertices.size() * sizeof(PreviewScreenVertex), nullptr));
    bridge->laser_index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    bridge->laser_indices.data(),
                    bridge->laser_indices.size() * sizeof(uint16_t), nullptr));
    auto& transforms = bridge->engine->getTransformManager();
    for (auto& entity : bridge->laser_entities) {
        entity = utils::EntityManager::get().create();
        transforms.create(entity);
        const auto result = filament::RenderableManager::Builder(1)
                .boundingBox({{-1.0f, -1.0f, -1.0f}, {1.0f, 2.0f, 1.0f}})
                .material(0, bridge->laser_material_instance)
                .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                        bridge->laser_vertex_buffer, bridge->laser_index_buffer,
                        0, static_cast<uint32_t>(bridge->laser_indices.size()))
                .priority(7)
                .culling(false)
                .castShadows(false)
                .receiveShadows(false)
                .build(*bridge->engine, entity);
        if (result != filament::RenderableManager::Builder::Success) {
            bridge_set_error(bridge, "Filament could not create controller laser renderable");
            return 0;
        }
        bridge->scene->addEntity(entity);
        bridge_set_renderable_visible(bridge, entity, false);
    }
    return 1;
}

int bridge_laser_set(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16, int visible) {
    if (!bridge || !bridge->engine || hand > 1 ||
            bridge->laser_entities[hand].isNull()) return 0;
    const auto entity = bridge->laser_entities[hand];
    if (!visible) {
        bridge_set_renderable_visible(bridge, entity, false);
        return 1;
    }
    if (!matrix16) return 0;
    auto& transforms = bridge->engine->getTransformManager();
    const auto instance = transforms.getInstance(entity);
    if (!instance.isValid()) return 0;
    const filament::math::mat4f matrix(
            matrix16[0], matrix16[1], matrix16[2], matrix16[3],
            matrix16[4], matrix16[5], matrix16[6], matrix16[7],
            matrix16[8], matrix16[9], matrix16[10], matrix16[11],
            matrix16[12], matrix16[13], matrix16[14], matrix16[15]);
    transforms.setTransform(instance, matrix);
    bridge_set_renderable_visible(bridge, entity, true);
    return 1;
}
