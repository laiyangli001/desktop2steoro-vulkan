from dataclasses import dataclass


GLTF_COLOR_SPACE_SRGB = "srgb"
GLTF_COLOR_SPACE_LINEAR = "linear"


@dataclass(frozen=True)
class MaterialTextureBinding:
    role: str
    material_key: str
    source_tex_field: str
    source_sampler_field: str
    opengl_uniform: str
    opengl_texture_unit: int
    d3d11_srv_slot: int
    color_space: str


# Shared glTF material contract for OpenGL and D3D11 renderers.
# glTF color textures are sRGB; data textures must stay linear.
GLTF_MATERIAL_TEXTURE_BINDINGS = (
    MaterialTextureBinding("base", "base_key", "tex_id", "base_sampler", "texture", 3, 0, GLTF_COLOR_SPACE_SRGB),
    MaterialTextureBinding("normal", "normal_key", "normal_tex_id", "normal_sampler", "normal_tex", 4, 3, GLTF_COLOR_SPACE_LINEAR),
    MaterialTextureBinding("occlusion", "occlusion_key", "occlusion_tex_id", "occlusion_sampler", "occlusion_tex", 5, 4, GLTF_COLOR_SPACE_LINEAR),
    MaterialTextureBinding("mr", "mr_key", "mr_tex_id", "mr_sampler", "mr_tex", 6, 5, GLTF_COLOR_SPACE_LINEAR),
    MaterialTextureBinding("emissive", "emissive_key", "emissive_tex_id", "emissive_sampler", "emissive_tex", 7, 6, GLTF_COLOR_SPACE_SRGB),
)

GLTF_TEXTURE_FIELDS = tuple(
    (binding.role, binding.source_tex_field, binding.source_sampler_field)
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
)

GLTF_COLOR_TEXTURE_KEYS = tuple(
    binding.material_key
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
    if binding.color_space == GLTF_COLOR_SPACE_SRGB
)

GLTF_DATA_TEXTURE_KEYS = tuple(
    binding.material_key
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
    if binding.color_space == GLTF_COLOR_SPACE_LINEAR
)