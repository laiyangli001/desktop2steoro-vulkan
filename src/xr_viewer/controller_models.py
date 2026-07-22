"""Controller asset inventory and profile selection shared by the OpenXR path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ControllerBrand:
    name: str
    root: Path
    left_glb: Path
    right_glb: Path
    offset: tuple[float, float, float]
    rotation_deg: float
    profile_id: str


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


@lru_cache(maxsize=32)
def controller_button_local_position(glb_path: str, button: str) -> tuple[float, float, float] | None:
    """Resolve a controller button node origin in model-local coordinates."""
    from .gltf.document import _load_gltf_document
    from .gltf.scene import _build_node_matrices

    try:
        document, _binary = _load_gltf_document(glb_path)
        world_matrices = _build_node_matrices(document)
    except (OSError, ValueError, TypeError):
        return None

    semantic = str(button).strip().lower()
    candidates = []
    for index, node in enumerate(document.get("nodes", [])):
        name = str(node.get("name") or "").strip().lower()
        if name in {semantic, f"right_{semantic}", f"{semantic}_mesh"} or name.endswith(f"_{semantic}"):
            if not any(marker in name for marker in ("_min", "_max", "_value")):
                candidates.append(index)
    if not candidates:
        return None
    position = world_matrices[candidates[0]][:3, 3]
    return tuple(float(value) for value in position)
