from __future__ import annotations

import ctypes
import json
import re
import struct
from pathlib import Path

import pytest

from xr_viewer.filament_vulkan_bridge import (
    FilamentBridgeError,
    FilamentVulkanBridge,
    _VulkanCreateInfo,
    _as_pointer_value,
    default_bridge_path,
)


def test_vulkan_create_info_has_stable_c_layout() -> None:
    assert ctypes.sizeof(_VulkanCreateInfo) == ctypes.sizeof(ctypes.c_void_p) * 3 + 8


def test_default_bridge_path_matches_platform() -> None:
    path = default_bridge_path()
    assert path.parent.name in {"windows", "linux", "macos"}
    assert path.name.startswith(("filament_bridge", "libfilament_bridge"))


def test_pointer_value_accepts_integer_and_c_void_p() -> None:
    assert _as_pointer_value(17) == 17
    assert _as_pointer_value(ctypes.c_void_p(23)) == 23


def test_missing_bridge_library_is_reported() -> None:
    with pytest.raises(FilamentBridgeError, match="unable to load"):
        FilamentVulkanBridge("missing-filament-bridge.dll")


def test_native_bridge_keeps_modular_resource_lifetimes_explicit() -> None:
    bridge_dir = (
        Path(__file__).resolve().parents[1] / "native/filament/bridge"
    )
    module_names = (
        "bridge_internal.h",
        "bridge_context.cpp",
        "bridge_context.h",
        "bridge_eye.cpp",
        "bridge_eye.h",
        "bridge_scene.cpp",
        "bridge_scene.h",
        "bridge_controller.cpp",
        "bridge_controller.h",
        "bridge_laser.cpp",
        "bridge_laser.h",
        "bridge_screen.cpp",
        "bridge_screen.h",
        "bridge_material.cpp",
        "bridge_material.h",
        "preview_bridge.cpp",
        "preview_bridge.h",
    )
    facade = (bridge_dir / "filament_bridge.cpp").read_text(encoding="utf-8")
    source = facade + "\n" + "\n".join(
        (bridge_dir / name).read_text(encoding="utf-8")
        for name in module_names
    )
    cmake = (bridge_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    public_header = (bridge_dir / "filament_bridge.h").read_text(
        encoding="utf-8"
    )
    abi_pattern = re.compile(
        r"\b(filament_(?:bridge|preview)_[a-z0-9_]+)\s*\("
    )

    assert set(abi_pattern.findall(public_header)) == set(
        abi_pattern.findall(facade)
    )
    assert all(
        not abi_pattern.search((bridge_dir / name).read_text(encoding="utf-8"))
        for name in module_names
    )
    assert "filament::" not in facade
    assert len(facade.splitlines()) < 400
    assert all(name in cmake for name in module_names if name.endswith(".cpp"))
    assert "filament::Renderer* renderer = nullptr;" in source
    assert "eye.renderer = bridge->engine->createRenderer();" in source
    assert "bridge->engine->destroy(eye.renderer);" in source
    assert "filament::View* laser_view = nullptr;" in source
    assert "eye.laser_view = bridge->engine->createView();" in source
    assert "eye.view->setVisibleLayers(0xff, 0x01);" in source
    assert "eye.laser_view->setVisibleLayers(0xff, 0x02);" in source
    assert "eye.laser_view->setPostProcessingEnabled(false);" in source
    assert "bridge->renderer->render(bridge->eyes[bridge->active_eye].laser_view);" in source
    assert "bridge_set_renderable_layer" in source
    assert ".exposure(target->brightness.scene_exposure_ev)" in source
    assert "scene_factor" not in source
    assert "return configure_color_pipeline_impl(preview) ? 1 : 0;" in source
    assert "VK_FORMAT_R8G8B8A8_SRGB" in source
    assert "Virtual screen requires VK_FORMAT_R8G8B8A8_SRGB" in source
    assert "Display-referred screen content bypasses the HDR scene view." in source
    assert "bridge_set_renderable_layer(bridge, bridge->screen_entity, 1, true);" in source
    assert "bool screen_in_scene = false;" in source
    assert "The sampler is required by the material" in source
    assert "filament_bridge_set_screen_ready_semaphore" in facade
    assert "pending_ready_semaphore" in source
    assert "screen_texture_cache" in source
    assert "bridge->engine->flushAndWait();" in source
    assert "diagnostic_frame_count < 8" in source
    assert "[FilamentBridge] acquired eye=" in source
    assert "filament_bridge_set_controller_visible" in facade
    assert "renderables.setLayerMask" in source
    assert "filament_bridge_set_controller_laser" in facade
    assert "D2S Controller Laser" in source
    assert 'parameter("laser_time"' in source
    assert "materialParams.laser_time * 0.4" in source
    assert "fract(uv.y + materialParams.laser_time * 0.4)" in source
    assert ".blending(filament::BlendingMode::OPAQUE)" in source
    assert ".depthWrite(true)" in source
    assert "materialParams_laser_time" not in source
    assert 'parameter("time"' not in source
    assert "float3(0.0, 0.4, 1.0)" in source
    assert "float3(1.0, 0.0, 0.0)" in source
    assert "std::array<PreviewScreenVertex, 8> laser_vertices" in source
    assert "std::array<uint16_t, 12> laser_indices" in source
    assert "controller_quaternion_slerp" in source
    assert "controller.button_values[5]" in source
    assert "controller loaded hand=%u animations=%zu" in source
    assert "kControllerValues" in source
    assert "getFirstEntityByName(value_name)" in source


@pytest.mark.parametrize("hand", ("left", "right"))
def test_pico_controller_glb_exposes_legacy_animation_triplets(hand: str) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/xr_viewer/controllers/PICO"
        / f"{hand}.glb"
    )
    payload = path.read_bytes()
    assert payload[:4] == b"glTF"
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(
        payload[20 : 20 + json_length].decode("utf-8").rstrip("\x00 ")
    )
    names = {str(node.get("name") or "") for node in document["nodes"]}
    value_names = {name for name in names if name.endswith("_value")}

    assert value_names
    assert all(
        value_name.removesuffix("_value") + suffix in names
        for value_name in value_names
        for suffix in ("_min", "_max")
    )
    assert any("trigger_pressed_value" in name for name in value_names)
    assert any("squeeze_pressed_value" in name for name in value_names)
    assert any("thumbstick_pressed_value" in name for name in value_names)
