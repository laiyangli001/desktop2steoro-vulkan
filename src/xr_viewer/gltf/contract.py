from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..material_contract import GLTF_COLOR_SPACE_LINEAR, GLTF_COLOR_SPACE_SRGB


VERTEX_FLOAT_COUNT = 10
TANGENT_FLOAT_COUNT = 4
D3D11_VERTEX_STRIDE_BYTES = VERTEX_FLOAT_COUNT * np.dtype(np.float32).itemsize
D3D11_VERTEX_OFFSETS_BYTES = (0, 12, 24, 32)
OPENGL_VERTEX_FORMAT = "3f 3f 2f 2f"

RenderPass = Literal["opaque", "mask", "transparent", "sky"]
TransparentSortPolicy = Literal["back_to_front"]
ColorSpace = Literal["srgb", "linear"]

_VALID_ALPHA_MODES = {"OPAQUE", "MASK", "BLEND"}
_VALID_RENDER_PASSES = {"opaque", "mask", "transparent", "sky"}
_DEFAULT_SAMPLER = (9729, 9987, 10497, 10497)
TRANSPARENT_SORT_POLICY: TransparentSortPolicy = "back_to_front"


@dataclass(frozen=True)
class TextureTransform:
    offset: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0


@dataclass(frozen=True)
class TextureBinding:
    image_id: int
    sampler: tuple[int, int, int, int] = _DEFAULT_SAMPLER
    texcoord: int = 0
    transform: TextureTransform = field(default_factory=TextureTransform)
    color_space: ColorSpace = GLTF_COLOR_SPACE_LINEAR


@dataclass(frozen=True)
class GltfMaterial:
    base_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    base_alpha: float = 1.0
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"] = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    texture_slots: Mapping[str, TextureBinding] = field(default_factory=dict)
    roughness: float = 1.0
    metallic: float = 1.0
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GltfPrimitive:
    vertices: np.ndarray
    tangent: np.ndarray
    indices: np.ndarray
    material: GltfMaterial
    node_name: str
    mesh_name: str
    world_bounds: tuple[np.ndarray, np.ndarray]
    render_pass: RenderPass
    primitive_mode: int = 4


RenderPlan = Mapping[RenderPass, tuple[int, ...]]


@dataclass(frozen=True)
class GltfScene:
    primitives: tuple[Mapping[str, Any], ...]
    textures: tuple[Any, ...]
    lights: tuple[Mapping[str, Any], ...]
    render_plan: RenderPlan
    transparent_sort: TransparentSortPolicy = TRANSPARENT_SORT_POLICY
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def apply_skybox_profile(primitives: Sequence[dict[str, Any]], profile: Mapping[str, Any]) -> int:
    """Apply explicit environment skybox settings to matching glTF primitives."""
    skybox = profile.get("skybox")
    if not isinstance(skybox, Mapping):
        return 0
    nodes = {str(name) for name in skybox.get("nodes", ()) if isinstance(name, str) and name}
    if not nodes:
        return 0
    force_opaque = bool(skybox.get("force_opaque", True))
    double_sided = bool(skybox.get("double_sided", True))
    matches = 0
    for primitive in primitives:
        if str(primitive.get("node_name") or "") not in nodes:
            continue
        primitive["sky_background"] = True
        primitive["render_pass"] = "sky"
        if force_opaque:
            primitive["alpha_mode"] = "OPAQUE"
        if double_sided:
            primitive["double_sided"] = True
        matches += 1
    return matches


def classify_render_pass(
    material: GltfMaterial,
    *,
    node_name: str = "",
    mesh_name: str = "",
    sky_background: bool = False,
) -> RenderPass:
    if sky_background:
        return "sky"
    if material.alpha_mode == "BLEND":
        return "transparent"
    if material.alpha_mode == "MASK":
        return "mask"
    return "opaque"


def validate_mesh_contract(vertices: np.ndarray, tangent: np.ndarray, indices: np.ndarray) -> None:
    if not isinstance(vertices, np.ndarray) or vertices.dtype != np.float32:
        raise ValueError("glTF vertices must be a float32 numpy array")
    if vertices.ndim != 2 or vertices.shape[1] != VERTEX_FLOAT_COUNT:
        raise ValueError(
            f"glTF vertices must have shape (N, {VERTEX_FLOAT_COUNT}); got {getattr(vertices, 'shape', None)}"
        )
    if not isinstance(tangent, np.ndarray) or tangent.dtype != np.float32:
        raise ValueError("glTF tangents must be a float32 numpy array")
    if tangent.ndim != 2 or tangent.shape != (vertices.shape[0], TANGENT_FLOAT_COUNT):
        raise ValueError(
            f"glTF tangents must have shape ({vertices.shape[0]}, {TANGENT_FLOAT_COUNT}); "
            f"got {getattr(tangent, 'shape', None)}"
        )
    if not isinstance(indices, np.ndarray) or indices.dtype != np.uint32 or indices.ndim != 1:
        raise ValueError("glTF indices must be a one-dimensional uint32 numpy array")


def build_primitive_contract(primitive: Mapping) -> GltfPrimitive:
    vertices = primitive.get("vertices")
    tangent = primitive.get("tangent")
    indices = primitive.get("indices")
    validate_mesh_contract(vertices, tangent, indices)

    material = primitive.get("material_contract")
    if not isinstance(material, GltfMaterial):
        raise ValueError("glTF primitive missing material_contract")

    node_name = str(primitive.get("node_name") or "")
    mesh_name = str(primitive.get("mesh_name") or "")
    world_bounds = (
        vertices[:, :3].min(axis=0).astype(np.float32),
        vertices[:, :3].max(axis=0).astype(np.float32),
    )
    render_pass = classify_render_pass(
        material,
        node_name=node_name,
        mesh_name=mesh_name,
        sky_background=bool(primitive.get("sky_background", False)),
    )
    return GltfPrimitive(
        vertices=vertices,
        tangent=tangent,
        indices=indices,
        material=material,
        node_name=node_name,
        mesh_name=mesh_name,
        world_bounds=world_bounds,
        render_pass=render_pass,
        primitive_mode=int(primitive.get("primitive_mode", 4)),
    )


def attach_primitive_contract(primitive: dict) -> GltfPrimitive:
    contract = build_primitive_contract(primitive)
    primitive["material_contract"] = contract.material
    primitive["gltf_primitive"] = contract
    primitive["render_pass"] = contract.render_pass
    primitive["world_bounds"] = contract.world_bounds
    return contract


def build_render_plan(primitives: Sequence[Mapping]) -> dict[RenderPass, tuple[int, ...]]:
    buckets: dict[RenderPass, list[int]] = {
        "sky": [],
        "opaque": [],
        "mask": [],
        "transparent": [],
    }
    for index, primitive in enumerate(primitives):
        buckets[render_pass_from_primitive(primitive)].append(index)
    return {render_pass: tuple(indices) for render_pass, indices in buckets.items()}


def primitive_sort_center(primitive: Mapping) -> np.ndarray:
    center = primitive.get("sort_center_local")
    if center is not None:
        return np.asarray(center, dtype=np.float32)
    world_bounds = primitive.get("world_bounds")
    if isinstance(world_bounds, tuple) and len(world_bounds) == 2:
        bounds_min = np.asarray(world_bounds[0], dtype=np.float32)
        bounds_max = np.asarray(world_bounds[1], dtype=np.float32)
        return ((bounds_min + bounds_max) * 0.5).astype(np.float32)
    vertices = primitive.get("vertices")
    if isinstance(vertices, np.ndarray) and vertices.ndim == 2 and vertices.shape[0] > 0 and vertices.shape[1] >= 3:
        return vertices[:, :3].mean(axis=0).astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def transparent_sort_key(
    primitive: Mapping,
    eye_position: Sequence[float],
    model_matrix: np.ndarray | None = None,
) -> float:
    center = primitive_sort_center(primitive)
    if model_matrix is not None:
        model = np.asarray(model_matrix, dtype=np.float32)
        local_center = np.array(
            [float(center[0]), float(center[1]), float(center[2]), 1.0],
            dtype=np.float32,
        )
        world_center = model @ local_center
        center = world_center[:3]
    eye = np.asarray(eye_position, dtype=np.float32)[:3]
    delta = center[:3] - eye
    return float(np.dot(delta, delta))


def sort_transparent_primitives(
    primitives: Sequence[Mapping],
    eye_position: Sequence[float],
    model_matrix: np.ndarray | None = None,
) -> list[Mapping]:
    return sorted(
        primitives,
        key=lambda primitive: transparent_sort_key(primitive, eye_position, model_matrix),
        reverse=True,
    )


def render_pass_from_primitive(primitive: Mapping) -> RenderPass:
    render_pass = primitive.get("render_pass")
    if render_pass in _VALID_RENDER_PASSES:
        return render_pass
    contract = primitive.get("gltf_primitive")
    if isinstance(contract, GltfPrimitive):
        return contract.render_pass
    material = primitive.get("material_contract")
    if not isinstance(material, GltfMaterial):
        raise ValueError("glTF primitive missing material_contract")
    return classify_render_pass(
        material,
        node_name=str(primitive.get("node_name") or ""),
        mesh_name=str(primitive.get("mesh_name") or ""),
        sky_background=bool(primitive.get("sky_background", False)),
    )


__all__ = [
    "ColorSpace",
    "D3D11_VERTEX_OFFSETS_BYTES",
    "D3D11_VERTEX_STRIDE_BYTES",
    "GltfMaterial",
    "GltfPrimitive",
    "GltfScene",
    "OPENGL_VERTEX_FORMAT",
    "RenderPass",
    "RenderPlan",
    "TANGENT_FLOAT_COUNT",
    "TRANSPARENT_SORT_POLICY",
    "TextureBinding",
    "TextureTransform",
    "VERTEX_FLOAT_COUNT",
    "attach_primitive_contract",
    "apply_skybox_profile",
    "build_primitive_contract",
    "build_render_plan",
    "classify_render_pass",
    "primitive_sort_center",
    "render_pass_from_primitive",
    "sort_transparent_primitives",
    "transparent_sort_key",
    "validate_mesh_contract",
    "GLTF_COLOR_SPACE_LINEAR",
    "GLTF_COLOR_SPACE_SRGB",
]
