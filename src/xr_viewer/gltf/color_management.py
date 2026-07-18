"""glTF color-space and output transform policy shared by render backends."""

from dataclasses import dataclass

from ..material_contract import (
    GLTF_COLOR_SPACE_LINEAR,
    GLTF_COLOR_SPACE_SRGB,
    GLTF_MATERIAL_TEXTURE_BINDINGS,
)


@dataclass(frozen=True)
class GltfColorManagementPolicy:
    """Renderer-facing color policy for glTF materials.

    glTF color textures are authored in sRGB, data textures are linear, and
    renderer lighting is evaluated in linear space before the backend writes
    gamma-encoded output.
    """

    color_texture_roles: tuple[str, ...]
    data_texture_roles: tuple[str, ...]
    shader_linear_space: str = GLTF_COLOR_SPACE_LINEAR
    srgb_decode_function: str = "gltfSrgbToLinear"
    tone_mapping: str = "reinhard"
    output_encode_function: str = "gltfLinearToOutput"
    default_output_gamma: float = 2.2
    default_exposure: float = 1.0


def _roles_for_color_space(color_space: str) -> tuple[str, ...]:
    return tuple(
        binding.role
        for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
        if binding.color_space == color_space
    )


DEFAULT_GLTF_COLOR_POLICY = GltfColorManagementPolicy(
    color_texture_roles=_roles_for_color_space(GLTF_COLOR_SPACE_SRGB),
    data_texture_roles=_roles_for_color_space(GLTF_COLOR_SPACE_LINEAR),
)


def color_space_for_texture_role(role: str) -> str:
    normalized = str(role)
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
        if binding.role == normalized:
            return binding.color_space
    raise KeyError(f"unknown glTF texture role: {role!r}")


def is_srgb_texture_role(role: str) -> bool:
    return color_space_for_texture_role(role) == GLTF_COLOR_SPACE_SRGB


def is_linear_texture_role(role: str) -> bool:
    return color_space_for_texture_role(role) == GLTF_COLOR_SPACE_LINEAR


def color_management_diagnostics(
    policy: GltfColorManagementPolicy = DEFAULT_GLTF_COLOR_POLICY,
) -> dict[str, object]:
    return {
        "colorTextureRoles": list(policy.color_texture_roles),
        "dataTextureRoles": list(policy.data_texture_roles),
        "shaderLinearSpace": policy.shader_linear_space,
        "srgbDecodeFunction": policy.srgb_decode_function,
        "toneMapping": policy.tone_mapping,
        "outputEncodeFunction": policy.output_encode_function,
        "defaultOutputGamma": float(policy.default_output_gamma),
        "defaultExposure": float(policy.default_exposure),
    }


__all__ = [
    "DEFAULT_GLTF_COLOR_POLICY",
    "GltfColorManagementPolicy",
    "color_management_diagnostics",
    "color_space_for_texture_role",
    "is_linear_texture_role",
    "is_srgb_texture_role",
]
