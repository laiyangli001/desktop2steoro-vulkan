#include "bridge_controller_guide.h"
#include "bridge_internal.h"

#include <cstring>
#include <utility>

void bridge_controller_guide_destroy(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return;
    if (!bridge->controller_guide_entity.isNull()) {
        if (bridge->scene) bridge->scene->remove(bridge->controller_guide_entity);
        bridge->engine->destroy(bridge->controller_guide_entity);
        bridge->controller_guide_entity = {};
    }
    if (bridge->controller_guide_vertex_buffer) {
        bridge->engine->destroy(bridge->controller_guide_vertex_buffer);
        bridge->controller_guide_vertex_buffer = nullptr;
    }
    if (bridge->controller_guide_index_buffer) {
        bridge->engine->destroy(bridge->controller_guide_index_buffer);
        bridge->controller_guide_index_buffer = nullptr;
    }
    if (bridge->controller_guide_material_instance) {
        bridge->engine->destroy(bridge->controller_guide_material_instance);
        bridge->controller_guide_material_instance = nullptr;
    }
    if (bridge->controller_guide_texture) {
        bridge->engine->destroy(bridge->controller_guide_texture);
        bridge->controller_guide_texture = nullptr;
    }
    if (bridge->controller_guide_material) {
        bridge->engine->destroy(bridge->controller_guide_material);
        bridge->controller_guide_material = nullptr;
    }
}

int bridge_controller_guide_set_texture(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height) {
    if (!bridge || !bridge->engine || !bridge->scene || !rgba ||
            width == 0 || height == 0) return 0;
    bridge_controller_guide_destroy(bridge);

    const char* shader = R"FILAMENT(
        void material(inout MaterialInputs material) {
            prepareMaterial(material);
            float4 guide = texture(materialParams_guideTexture, getUV0());
            // Filament's transparent blend mode expects premultiplied alpha.
            // The source texture deliberately keeps white RGB in transparent
            // texels for clean filtered edges, so premultiply after sampling.
            guide.rgb *= guide.a;
            material.baseColor = guide;
        }
    )FILAMENT";
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S Controller Guide")
            .material(shader)
            .require(filament::VertexAttribute::UV0)
            .parameter("guideTexture", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
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
        bridge_set_error(bridge, "Filament could not build controller guide material");
        return 0;
    }
    bridge->controller_guide_material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*bridge->engine);
    if (!bridge->controller_guide_material) {
        bridge_set_error(bridge, "Filament could not create controller guide material");
        return 0;
    }
    bridge->controller_guide_material_instance =
            bridge->controller_guide_material->createInstance();
    bridge->controller_guide_texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            .format(filament::Texture::InternalFormat::SRGB8_A8)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .build(*bridge->engine);
    if (!bridge->controller_guide_material_instance ||
            !bridge->controller_guide_texture) {
        bridge_set_error(bridge, "Filament could not create controller guide texture");
        return 0;
    }

    const size_t byte_count = static_cast<size_t>(width) * height * 4u;
    auto* pixels = new uint8_t[byte_count];
    std::memcpy(pixels, rgba, byte_count);
    filament::Texture::PixelBufferDescriptor descriptor(
            pixels, byte_count,
            filament::Texture::Format::RGBA,
            filament::Texture::Type::UBYTE,
            [](void* buffer, size_t, void*) {
                delete[] static_cast<uint8_t*>(buffer);
            });
    bridge->controller_guide_texture->setImage(
            *bridge->engine, 0, std::move(descriptor));
    bridge->controller_guide_material_instance->setParameter(
            "guideTexture", bridge->controller_guide_texture,
            bridge->controller_guide_texture_sampler);

    bridge->controller_guide_vertices = {{
            {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f}},
            {{ 0.5f, -0.5f, 0.0f}, {1.0f, 0.0f}},
            {{-0.5f,  0.5f, 0.0f}, {0.0f, 1.0f}},
            {{ 0.5f,  0.5f, 0.0f}, {1.0f, 1.0f}},
    }};
    bridge->controller_guide_indices = {{0, 1, 2, 1, 3, 2}};
    bridge->controller_guide_vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(4).bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(PreviewScreenVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(PreviewScreenVertex))
            .build(*bridge->engine);
    bridge->controller_guide_index_buffer = filament::IndexBuffer::Builder()
            .indexCount(6)
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*bridge->engine);
    if (!bridge->controller_guide_vertex_buffer ||
            !bridge->controller_guide_index_buffer) {
        bridge_set_error(bridge, "Filament could not create controller guide geometry");
        return 0;
    }
    bridge->controller_guide_vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    bridge->controller_guide_vertices.data(),
                    bridge->controller_guide_vertices.size() * sizeof(PreviewScreenVertex),
                    nullptr));
    bridge->controller_guide_index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    bridge->controller_guide_indices.data(),
                    bridge->controller_guide_indices.size() * sizeof(uint16_t), nullptr));

    bridge->controller_guide_entity = utils::EntityManager::get().create();
    bridge->engine->getTransformManager().create(bridge->controller_guide_entity);
    const auto result = filament::RenderableManager::Builder(1)
            .boundingBox({{-1.0f, -1.0f, -0.1f}, {1.0f, 1.0f, 0.1f}})
            .material(0, bridge->controller_guide_material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    bridge->controller_guide_vertex_buffer,
                    bridge->controller_guide_index_buffer, 0, 6)
            .priority(7).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, bridge->controller_guide_entity);
    if (result != filament::RenderableManager::Builder::Success) {
        bridge_set_error(bridge, "Filament could not create controller guide renderable");
        return 0;
    }
    bridge->scene->addEntity(bridge->controller_guide_entity);
    bridge_set_renderable_layer(bridge, bridge->controller_guide_entity, 1, false);
    return 1;
}

int bridge_controller_guide_set(
        FilamentBridge* bridge, const float* matrix16, int visible) {
    if (!bridge || !bridge->engine || bridge->controller_guide_entity.isNull()) return 0;
    if (!visible) {
        bridge_set_renderable_layer(bridge, bridge->controller_guide_entity, 1, false);
        return 1;
    }
    if (!matrix16) return 0;
    auto& transforms = bridge->engine->getTransformManager();
    const auto instance = transforms.getInstance(bridge->controller_guide_entity);
    if (!instance.isValid()) return 0;
    const filament::math::mat4f matrix(
            matrix16[0], matrix16[1], matrix16[2], matrix16[3],
            matrix16[4], matrix16[5], matrix16[6], matrix16[7],
            matrix16[8], matrix16[9], matrix16[10], matrix16[11],
            matrix16[12], matrix16[13], matrix16[14], matrix16[15]);
    transforms.setTransform(instance, matrix);
    bridge_set_renderable_layer(bridge, bridge->controller_guide_entity, 1, true);
    return 1;
}
