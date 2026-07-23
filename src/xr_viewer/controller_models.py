"""Controller asset inventory and profile selection shared by the OpenXR path."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ControllerBrand:
    name: str
    root: Path
    left_glb: Path
    right_glb: Path
    offset: tuple[float, float, float]
    rotation_deg: float
    profile_id: str
    ambient_light_multiplier: float


def _vector3(value, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return default
    return tuple(float(item) for item in value[:3])


def discover_controller_brands(root: str | Path) -> dict[str, ControllerBrand]:
    """Discover every complete brand directory under controllers/."""
    base = Path(root)
    result: dict[str, ControllerBrand] = {}
    if not base.is_dir():
        return result
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        left_glb = directory / "left.glb"
        right_glb = directory / "right.glb"
        if not left_glb.is_file() or not right_glb.is_file():
            continue
        profile = {}
        profile_path = directory / "profile.json"
        if profile_path.is_file():
            try:
                profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                profile = {}
        overrides = profile.get("overrides", {}) if isinstance(profile, dict) else {}
        if not isinstance(overrides, dict):
            overrides = {}
        result[directory.name] = ControllerBrand(
            name=directory.name,
            root=directory,
            left_glb=left_glb,
            right_glb=right_glb,
            offset=_vector3(overrides.get("model_offset")),
            rotation_deg=float(overrides.get("model_rotation_deg", 0.0)),
            profile_id=str(profile.get("profileId", directory.name))
            if isinstance(profile, dict)
            else directory.name,
            ambient_light_multiplier=max(
                0.0, float(overrides.get("ambient_light_multiplier", 1.0))
            ),
        )
    return result


def select_controller_brand(
    brands: dict[str, ControllerBrand], requested: str | None
) -> ControllerBrand | None:
    if not brands:
        return None
    if requested and requested in brands:
        return brands[requested]
    return brands[sorted(brands)[0]]


def _load_controller_document(path: str | Path) -> dict:
    """Read only the glTF JSON needed for controller node transforms."""
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        return json.loads(data.decode("utf-8-sig"))
    if len(data) < 20:
        raise ValueError("truncated GLB header")
    _magic, version, byte_length = struct.unpack_from("<4sII", data, 0)
    if version != 2 or byte_length > len(data):
        raise ValueError("invalid GLB header")
    offset = 12
    while offset + 8 <= byte_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > byte_length:
            raise ValueError("truncated GLB chunk")
        if chunk_type == 0x4E4F534A:
            return json.loads(data[offset:chunk_end].decode("utf-8").rstrip("\x00 "))
        offset = chunk_end
    raise ValueError("GLB JSON chunk is missing")


def _controller_node_local_matrix(node: dict) -> np.ndarray:
    matrix = node.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        return np.asarray(matrix, dtype=np.float64).reshape((4, 4)).T
    translation = node.get("translation", (0.0, 0.0, 0.0))
    x, y, z, w = (float(value) for value in node.get("rotation", (0, 0, 0, 1)))
    scale = node.get("scale", (1.0, 1.0, 1.0))
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0),
            (0, 0, 0, 1),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return transform @ rotation @ np.diag((*scale, 1.0))


def _controller_node_world_matrices(document: dict) -> list[np.ndarray]:
    nodes = document.get("nodes", [])
    local = [_controller_node_local_matrix(node) for node in nodes]
    parent = [-1] * len(nodes)
    for parent_index, node in enumerate(nodes):
        for child_index in node.get("children", []):
            if isinstance(child_index, int) and 0 <= child_index < len(nodes):
                parent[child_index] = parent_index
    world: list[np.ndarray | None] = [None] * len(nodes)

    def resolve(index: int) -> np.ndarray:
        if world[index] is None:
            parent_index = parent[index]
            world[index] = (
                local[index]
                if parent_index < 0
                else resolve(parent_index) @ local[index]
            )
        return world[index]

    return [resolve(index) for index in range(len(nodes))]


@lru_cache(maxsize=32)
def controller_button_local_position(glb_path: str, button: str) -> tuple[float, float, float] | None:
    """Resolve a controller button node origin in model-local coordinates."""
    try:
        document = _load_controller_document(glb_path)
        world_matrices = _controller_node_world_matrices(document)
    except (OSError, ValueError, TypeError, RecursionError):
        return None

    semantic = str(button).strip().lower().replace("-", "_").replace(" ", "_")
    preferred_names = (
        f"{semantic}_pressed_value",
        f"{semantic}_value",
        semantic,
        f"right_{semantic}",
        f"{semantic}_mesh",
        f"{semantic}_pressed_min",
        f"{semantic}_min",
    )
    rank_by_name = {name: rank for rank, name in enumerate(preferred_names)}
    candidates: list[tuple[int, int]] = []
    for index, node in enumerate(document.get("nodes", [])):
        name = (
            str(node.get("name") or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        rank = rank_by_name.get(name)
        if rank is None:
            for suffix_rank, suffix in enumerate(preferred_names):
                if name.endswith(f"_{suffix}"):
                    rank = len(preferred_names) + suffix_rank
                    break
        if rank is not None:
            candidates.append((rank, index))
    if not candidates:
        return None
    _rank, selected_index = min(candidates)
    position = world_matrices[selected_index][:3, 3]
    return tuple(float(value) for value in position)
