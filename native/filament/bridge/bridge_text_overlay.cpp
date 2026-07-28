#include "bridge_text_overlay.h"

#include "bridge_internal.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace {

constexpr uint32_t kMaxGlyphsPerPage = 4096;
constexpr uint32_t kMaxVertices = kMaxGlyphsPerPage * 4;
constexpr uint32_t kMaxIndices = kMaxGlyphsPerPage * 6;

const char* kMsdfShader = R"FILAMENT(
    // Keep the derivative-based MSDF transition neutral. Artificially
    // increasing this range makes small CJK counters merge in the headset.
    const float kMinimumScreenPxRange = 1.0;
    const float kEdgeSharpness = 1.0;

    float median(float r, float g, float b) {
        return max(min(r, g), min(max(r, g), b));
    }

    void material(inout MaterialInputs material) {
        prepareMaterial(material);
        float3 msdf_sample = texture(materialParams_atlas, getUV0()).rgb;
        float signed_distance = median(
                msdf_sample.r, msdf_sample.g, msdf_sample.b) - 0.5;
        // Convert the atlas distance range into screen pixels. A plain
        // fwidth(distance) becomes unstable when the VR overlay is small.
        float2 atlas_size = float2(2048.0, 2048.0);
        float2 unit_range = float2(4.0) / atlas_size;
        float2 screen_tex_size = 1.0 / max(fwidth(getUV0()), float2(0.000001));
        float screen_px_range = max(
                0.5 * dot(unit_range, screen_tex_size),
                kMinimumScreenPxRange) * kEdgeSharpness;
        float coverage = clamp(
                screen_px_range * signed_distance + 0.5, 0.0, 1.0);
        float4 vertex_color = getColor();
        float alpha = vertex_color.a * coverage;
        material.baseColor = float4(vertex_color.rgb, alpha);
    }
)FILAMENT";

void destroy_page(FilamentBridge* bridge, MsdfTextPage& page) {
    if (!bridge || !bridge->engine) return;
    if (!page.entity.isNull()) {
        if (bridge->scene) bridge->scene->remove(page.entity);
        bridge->engine->destroy(page.entity);
        page.entity = {};
    }
    if (page.vertex_buffer) {
        bridge->engine->destroy(page.vertex_buffer);
        page.vertex_buffer = nullptr;
    }
    if (page.index_buffer) {
        bridge->engine->destroy(page.index_buffer);
        page.index_buffer = nullptr;
    }
    if (page.material_instance) {
        bridge->engine->destroy(page.material_instance);
        page.material_instance = nullptr;
    }
    if (page.texture) {
        bridge->engine->destroy(page.texture);
        page.texture = nullptr;
    }
    if (page.material) {
        bridge->engine->destroy(page.material);
        page.material = nullptr;
    }
    page.vertices.clear();
    page.indices.clear();
}

bool create_page_geometry(FilamentBridge* bridge, MsdfTextPage& page) {
    page.vertices.assign(kMaxVertices, MsdfTextVertex{});
    page.indices.assign(kMaxIndices, 0);
    for (uint32_t glyph = 0; glyph < kMaxGlyphsPerPage; ++glyph) {
        const uint32_t vertex = glyph * 4;
        const uint32_t index = glyph * 6;
        page.indices[index + 0] = static_cast<uint16_t>(vertex + 0);
        page.indices[index + 1] = static_cast<uint16_t>(vertex + 1);
        page.indices[index + 2] = static_cast<uint16_t>(vertex + 2);
        page.indices[index + 3] = static_cast<uint16_t>(vertex + 1);
        page.indices[index + 4] = static_cast<uint16_t>(vertex + 3);
        page.indices[index + 5] = static_cast<uint16_t>(vertex + 2);
    }
    page.vertex_buffer = filament::VertexBuffer::Builder()
            .vertexCount(kMaxVertices).bufferCount(1)
            .attribute(filament::VertexAttribute::POSITION, 0,
                    filament::VertexBuffer::AttributeType::FLOAT3,
                    0, sizeof(MsdfTextVertex))
            .attribute(filament::VertexAttribute::UV0, 0,
                    filament::VertexBuffer::AttributeType::FLOAT2,
                    sizeof(float) * 3, sizeof(MsdfTextVertex))
            .attribute(filament::VertexAttribute::COLOR, 0,
                    filament::VertexBuffer::AttributeType::FLOAT4,
                    sizeof(float) * 5, sizeof(MsdfTextVertex))
            .build(*bridge->engine);
    page.index_buffer = filament::IndexBuffer::Builder()
            .indexCount(kMaxIndices)
            .bufferType(filament::IndexBuffer::IndexType::USHORT)
            .build(*bridge->engine);
    if (!page.vertex_buffer || !page.index_buffer) return false;
    page.vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    page.vertices.data(),
                    page.vertices.size() * sizeof(MsdfTextVertex), nullptr));
    page.index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    page.indices.data(),
                    page.indices.size() * sizeof(uint16_t), nullptr));
    page.entity = utils::EntityManager::get().create();
    bridge->engine->getTransformManager().create(page.entity);
    const auto result = filament::RenderableManager::Builder(1)
            .boundingBox({{-10000.0f, -10000.0f, -10000.0f},
                          {10000.0f, 10000.0f, 10000.0f}})
            .material(0, page.material_instance)
            .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES,
                    page.vertex_buffer, page.index_buffer, 0, kMaxIndices)
            .priority(7).culling(false).castShadows(false).receiveShadows(false)
            .build(*bridge->engine, page.entity);
    if (result != filament::RenderableManager::Builder::Success) return false;
    bridge->scene->addEntity(page.entity);
    bridge_set_renderable_layer(bridge, page.entity, 1, false);
    return true;
}

bool ensure_page_material(FilamentBridge* bridge, MsdfTextPage& page) {
    if (page.material_instance && page.vertex_buffer && page.index_buffer) return true;
    filamat::MaterialBuilder::init();
    filamat::MaterialBuilder builder;
    builder.name("D2S MSDF Text")
            .material(kMsdfShader)
            .require(filament::VertexAttribute::UV0)
            .require(filament::VertexAttribute::COLOR)
            .parameter("atlas", filamat::MaterialBuilder::SamplerType::SAMPLER_2D)
            .shading(filament::Shading::UNLIT)
            .materialDomain(filament::MaterialDomain::SURFACE)
            .blending(filament::BlendingMode::TRANSPARENT)
            .culling(filament::backend::CullingMode::NONE)
            .depthWrite(false).depthCulling(false)
            .targetApi(filamat::MaterialBuilder::TargetApi::ALL)
            .platform(filamat::MaterialBuilder::Platform::ALL);
    const filamat::Package package = builder.build(bridge->engine->getJobSystem());
    if (!package.isValid()) {
        bridge_set_error(bridge, "Filament could not build MSDF text material");
        return false;
    }
    page.material = filament::Material::Builder()
            .package(package.getData(), package.getSize())
            .build(*bridge->engine);
    page.material_instance = page.material ? page.material->createInstance() : nullptr;
    if (!page.material || !page.material_instance) {
        bridge_set_error(bridge, "Filament could not create MSDF text material");
        return false;
    }
    return create_page_geometry(bridge, page);
}

}  // namespace

void bridge_text_overlay_destroy(FilamentBridge* bridge) {
    if (!bridge) return;
    for (auto& page : bridge->text_pages) destroy_page(bridge, page);
}

int bridge_text_overlay_set_page_texture(
        FilamentBridge* bridge, uint32_t page_index,
        const uint8_t* rgba, uint32_t width, uint32_t height) {
    if (!bridge || !bridge->engine || !bridge->scene || !rgba ||
            page_index >= bridge->text_pages.size() || width == 0 || height == 0) {
        return 0;
    }
    auto& page = bridge->text_pages[page_index];
    if (!ensure_page_material(bridge, page)) return 0;
    if (page.texture) bridge->engine->destroy(page.texture);
    page.texture = filament::Texture::Builder()
            .width(width).height(height).levels(1)
            .format(filament::Texture::InternalFormat::RGBA8)
            .sampler(filament::Texture::Sampler::SAMPLER_2D)
            .build(*bridge->engine);
    if (!page.texture) {
        bridge_set_error(bridge, "Filament could not create MSDF text atlas texture");
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
    page.texture->setImage(*bridge->engine, 0, std::move(descriptor));
    page.sampler.setMinFilter(filament::TextureSampler::MinFilter::LINEAR);
    page.sampler.setMagFilter(filament::TextureSampler::MagFilter::LINEAR);
    page.sampler.setWrapModeS(filament::TextureSampler::WrapMode::CLAMP_TO_EDGE);
    page.sampler.setWrapModeT(filament::TextureSampler::WrapMode::CLAMP_TO_EDGE);
    page.material_instance->setParameter("atlas", page.texture, page.sampler);
    return 1;
}

int bridge_text_overlay_set_page_vertices(
        FilamentBridge* bridge, uint32_t page_index,
        const float* vertices, uint32_t vertex_count,
        const uint16_t* indices, uint32_t index_count, int visible) {
    if (!bridge || !bridge->engine || page_index >= bridge->text_pages.size() ||
            !vertices || !indices || vertex_count > kMaxVertices ||
            index_count > kMaxIndices) return 0;
    auto& page = bridge->text_pages[page_index];
    if (!page.vertex_buffer || !page.index_buffer || page.entity.isNull()) return 0;
    std::fill(page.vertices.begin(), page.vertices.end(), MsdfTextVertex{});
    const auto* source = reinterpret_cast<const MsdfTextVertex*>(vertices);
    std::copy_n(source, vertex_count, page.vertices.begin());
    std::copy_n(indices, index_count, page.indices.begin());
    page.vertex_buffer->setBufferAt(*bridge->engine, 0,
            filament::VertexBuffer::BufferDescriptor(
                    page.vertices.data(),
                    page.vertices.size() * sizeof(MsdfTextVertex), nullptr));
    page.index_buffer->setBuffer(*bridge->engine,
            filament::IndexBuffer::BufferDescriptor(
                    page.indices.data(),
                    page.indices.size() * sizeof(uint16_t), nullptr));
    bridge_set_renderable_layer(bridge, page.entity, 1, visible != 0);
    return 1;
}
