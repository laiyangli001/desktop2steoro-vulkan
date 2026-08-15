#include "bridge_material_provider.h"

#include <filamat/MaterialBuilder.h>
#include <filament/Engine.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using filament::Engine;
using filament::Material;
using filament::MaterialInstance;
using filament::VertexAttribute;
using filament::gltfio::AlphaMode;
using filament::gltfio::MaterialKey;
using filament::gltfio::MaterialProvider;
using filament::gltfio::UvMap;
using filamat::MaterialBuilder;

constexpr char kReflectivePrefix[] = "D2S_REFLECTIVE__";

bool is_reflective_label(const char* label) {
    return label && std::strncmp(
            label, kReflectivePrefix, sizeof(kReflectivePrefix) - 1) == 0;
}

struct CachedMaterial {
    MaterialKey key{};
    UvMap uvmap{};
    Material* material = nullptr;
};

class BridgeMaterialProvider final : public MaterialProvider {
public:
    explicit BridgeMaterialProvider(Engine* engine)
            : engine_(engine), fallback_(filament::gltfio::createJitShaderProvider(engine)) {}

    ~BridgeMaterialProvider() override {
        delete fallback_;
    }

    MaterialInstance* createMaterialInstance(
            MaterialKey* config, UvMap* uvmap, const char* label,
            const char* extras) override {
        if (!is_reflective_label(label)) {
            return fallback_->createMaterialInstance(config, uvmap, label, extras);
        }
        auto* material = get_reflective_material(config, uvmap, label);
        return material ? material->createInstance(label) : nullptr;
    }

    Material* getMaterial(
            MaterialKey* config, UvMap* uvmap, const char* label) override {
        if (!is_reflective_label(label)) {
            return fallback_->getMaterial(config, uvmap, label);
        }
        return get_reflective_material(config, uvmap, label);
    }

    const Material* const* getMaterials() const noexcept override {
        material_view_.clear();
        const auto* fallback_materials = fallback_->getMaterials();
        const size_t fallback_count = fallback_->getMaterialsCount();
        if (fallback_count) {
            material_view_.insert(
                    material_view_.end(), fallback_materials,
                    fallback_materials + fallback_count);
        }
        for (const auto& entry : cache_) {
            material_view_.push_back(entry.material);
        }
        return material_view_.data();
    }

    size_t getMaterialsCount() const noexcept override {
        return fallback_->getMaterialsCount() + cache_.size();
    }

    void destroyMaterials() override {
        for (auto& entry : cache_) {
            engine_->destroy(entry.material);
        }
        cache_.clear();
        material_view_.clear();
        fallback_->destroyMaterials();
    }

    bool needsDummyData(VertexAttribute attrib) const noexcept override {
        return fallback_->needsDummyData(attrib);
    }

private:
    Material* get_reflective_material(
            MaterialKey* config, UvMap* uvmap, const char* label) {
        filament::gltfio::constrainMaterial(config, uvmap);
        for (const auto& entry : cache_) {
            if (entry.key == *config && entry.uvmap == *uvmap) {
                return entry.material;
            }
        }

        std::string shader = R"SHADER(
            void material(inout MaterialInputs material) {
                prepareMaterial(material);
                material.baseColor = materialParams.baseColorFactor;
        )SHADER";
        if (config->hasBaseColorTexture) {
            shader += "highp float2 baseColorUV = ${color};\n";
            if (config->hasTextureTransforms) {
                shader += "baseColorUV = (vec3(baseColorUV, 1.0) * "
                          "materialParams.baseColorUvMatrix).xy;\n";
            }
            shader += "material.baseColor *= texture(materialParams_baseColorMap, "
                      "baseColorUV);\n";
        }
        if (config->hasVertexColors) {
            shader += "material.baseColor *= getColor();\n";
        }
        if (config->alphaMode == AlphaMode::BLEND) {
            shader += "material.baseColor.rgb *= material.baseColor.a;\n";
        }
        shader += R"SHADER(
                material.metallic = materialParams.metallicFactor;
                material.roughness = materialParams.roughnessFactor;
                vec3 bakedBaseline = material.baseColor.rgb
                        * materialParams.emissiveFactor;
                vec3 screenAreaLight = vec3(0.0);
                vec4 screenState = getMaterialGlobal0();
                if (screenState.w > 0.5) {
                    vec4 screenNormalIntensity = getMaterialGlobal1();
                    vec4 screenColorHalfWidth = getMaterialGlobal2();
                    vec4 screenSize = getMaterialGlobal3();
                    vec3 screenToFragment = getUserWorldPosition()
                            - screenState.xyz;
                    float distanceToScreen = length(screenToFragment);
                    vec3 lightDirection = screenToFragment
                            / max(distanceToScreen, 0.001);
                    float front = smoothstep(0.0, 0.3, dot(
                            screenNormalIntensity.xyz, lightDirection));
                    vec3 worldNormal = getWorldNormalVector();
                    float normalResponse = max(dot(
                            worldNormal, -lightDirection), 0.0);
                    float halfWidth = max(screenColorHalfWidth.w, 0.001);
                    float halfHeight = max(screenSize.x, 0.001);
                    float halfDiagonal = length(vec2(halfWidth, halfHeight));
                    float broadRadius = max(halfDiagonal * 2.0, 0.50);
                    float attenuation = (broadRadius * broadRadius)
                            / (distanceToScreen * distanceToScreen
                            + broadRadius * broadRadius);
                    float area = 4.0 * halfWidth * halfHeight;
                    float nearRadius = max(halfDiagonal * 0.5, 0.10);
                    float areaResponse = area / (3.14159265 * max(
                            distanceToScreen * distanceToScreen,
                            nearRadius * nearRadius));
                    float haloFree = smoothstep(
                            max(halfDiagonal * 0.35, 0.75),
                            max(halfDiagonal * 0.95, 1.75),
                            distanceToScreen);
                    screenAreaLight = material.baseColor.rgb
                            * screenColorHalfWidth.rgb
                            * screenNormalIntensity.w
                            * front * normalResponse * attenuation
                            * areaResponse * haloFree;
                }
                material.emissive = vec4(bakedBaseline + screenAreaLight, 0.0);
            }
        )SHADER";
        filament::gltfio::processShaderString(&shader, *uvmap, *config);

        MaterialBuilder builder;
        builder.name(label)
                .flipUV(false)
                .shading(MaterialBuilder::Shading::LIT)
                .doubleSided(config->doubleSided)
                .transparencyMode(config->doubleSided
                        ? MaterialBuilder::TransparencyMode::TWO_PASSES_TWO_SIDES
                        : MaterialBuilder::TransparencyMode::DEFAULT)
                .targetApi(filamat::targetApiFromBackend(engine_->getBackend()))
                .stereoscopicType(engine_->getConfig().stereoscopicType)
                .stereoscopicEyeCount(engine_->getConfig().stereoscopicEyeCount)
                .optimization(MaterialBuilder::Optimization::NONE)
                .material(shader.c_str())
                .parameter("baseColorFactor", MaterialBuilder::UniformType::FLOAT4)
                .parameter("metallicFactor", MaterialBuilder::UniformType::FLOAT)
                .parameter("roughnessFactor", MaterialBuilder::UniformType::FLOAT)
                .parameter("emissiveFactor", MaterialBuilder::UniformType::FLOAT3)
                .parameter("normalScale", MaterialBuilder::UniformType::FLOAT)
                .parameter("aoStrength", MaterialBuilder::UniformType::FLOAT);

        const int uv_sets = filament::gltfio::getNumUvSets(*uvmap);
        if (uv_sets > 0) builder.require(VertexAttribute::UV0);
        if (uv_sets > 1) builder.require(VertexAttribute::UV1);
        if (config->hasVertexColors) builder.require(VertexAttribute::COLOR);
        if (config->hasBaseColorTexture) {
            builder.parameter("baseColorMap", MaterialBuilder::SamplerType::SAMPLER_2D);
            if (config->hasTextureTransforms) {
                builder.parameter("baseColorUvMatrix", MaterialBuilder::UniformType::MAT3,
                        MaterialBuilder::ParameterPrecision::HIGH);
            }
        }
        if (config->hasEmissiveTexture) {
            builder.parameter("emissiveMap", MaterialBuilder::SamplerType::SAMPLER_2D);
            if (config->hasTextureTransforms) {
                builder.parameter("emissiveUvMatrix", MaterialBuilder::UniformType::MAT3,
                        MaterialBuilder::ParameterPrecision::HIGH);
            }
        }
        switch (config->alphaMode) {
            case AlphaMode::MASK:
                builder.blending(MaterialBuilder::BlendingMode::MASKED);
                break;
            case AlphaMode::BLEND:
                builder.blending(MaterialBuilder::BlendingMode::FADE);
                break;
            default:
                builder.blending(MaterialBuilder::BlendingMode::OPAQUE);
                break;
        }

        auto package = builder.build(engine_->getJobSystem());
        auto* material = Material::Builder()
                .package(package.getData(), package.getSize())
                .build(*engine_);
        if (material) {
            cache_.push_back(CachedMaterial{*config, *uvmap, material});
            std::fprintf(stderr,
                    "[FilamentBridge] reflective room material active: "
                    "label=%s base_texture=%d emissive_texture=%d\n",
                    label ? label : "", config->hasBaseColorTexture ? 1 : 0,
                    config->hasEmissiveTexture ? 1 : 0);
        }
        return material;
    }

    Engine* engine_ = nullptr;
    MaterialProvider* fallback_ = nullptr;
    std::vector<CachedMaterial> cache_;
    mutable std::vector<const Material*> material_view_;
};

} // namespace

MaterialProvider* bridge_create_material_provider(Engine* engine) {
    return new BridgeMaterialProvider(engine);
}
