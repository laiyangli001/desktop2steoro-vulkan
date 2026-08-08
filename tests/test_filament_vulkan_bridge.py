from __future__ import annotations

import ctypes
import json
import re
import struct
import subprocess
import threading
from pathlib import Path

import pytest

from xr_viewer.filament_vulkan_bridge import (
    FilamentBridgeError,
    FilamentVulkanBridge,
    _FilamentLightingConfig,
    _VulkanCreateInfo,
    _as_pointer_value,
    default_bridge_path,
)


def test_vulkan_create_info_has_stable_c_layout() -> None:
    assert ctypes.sizeof(_VulkanCreateInfo) == ctypes.sizeof(ctypes.c_void_p) * 3 + 8
    assert ctypes.sizeof(_FilamentLightingConfig) == 112


def test_default_bridge_path_matches_platform() -> None:
    path = default_bridge_path()
    assert path.parent.name in {"windows", "linux", "macos"}
    assert path.name.startswith(("filament_bridge", "libfilament_bridge"))


def test_remote_filament_build_enables_multiview_without_stale_sdk_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github/workflows/filament-bridge.yml"
    ).read_text(encoding="utf-8")
    patch = (
        root / "native/filament/patches/apply_d2s_vulkan_external_image.py"
    ).read_text(encoding="utf-8")
    manifest = json.loads(
        (root / "native/filament/version.json").read_text(encoding="utf-8")
    )
    cmake = (root / "native/filament/bridge/CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert manifest["version"] == "1.75.0"
    assert manifest["source"]["ref"] == "v1.75.0"
    assert all("v1.75.0" in entry["asset"] for entry in manifest["platforms"].values())
    assert "google/filament/v1.75.0/libs/bluevk" in cmake
    assert "#if defined(VK_EXT_depth_clamp_control)" in cmake
    assert "VK_EXT_depth_clamp_control) || defined(VK_EXT_shader_object" in cmake
    assert workflow.count("-DFILAMENT_ENABLE_MULTIVIEW=ON") == 2
    assert workflow.count(
        "hashFiles('native/filament/version.json', "
        "'native/filament/patches/**', '.github/workflows/filament-bridge.yml')"
    ) == 2
    assert "[D2S stereo trace] renderer" in patch
    assert "d2sStereoTraceLogged[2]" in patch
    assert "variant.hasStereo()" in patch
    assert "[D2S stereo trace] renderPass" in patch
    assert "config.viewCount, subpassViewMask" in patch
    assert "[D2S stereo trace] program" in patch
    assert "builder.isMultiview()" in patch
    assert "words[word] == 4440u" in patch
    assert "[D2S stereo trace] draw viewCount=%u %s" in patch
    assert "rt->getRenderPassKey().viewCount" in patch
    assert "getViewType(getSamplerTypeFromDepth(depth))" in patch


def test_pointer_value_accepts_integer_and_c_void_p() -> None:
    assert _as_pointer_value(17) == 17
    assert _as_pointer_value(ctypes.c_void_p(23)) == 23


def test_missing_bridge_library_is_reported() -> None:
    with pytest.raises(FilamentBridgeError, match="unable to load"):
        FilamentVulkanBridge("missing-filament-bridge.dll")


def test_bridge_rejects_c_abi_calls_from_non_owner_thread() -> None:
    bridge = object.__new__(FilamentVulkanBridge)
    bridge._handle = ctypes.c_void_p(1)
    bridge._owner_thread_id = threading.get_ident() + 1

    with pytest.raises(FilamentBridgeError, match="Presenter owner thread"):
        bridge._ensure_loaded()


def test_native_bridge_keeps_modular_resource_lifetimes_explicit() -> None:
    bridge_dir = (
        Path(__file__).resolve().parents[1] / "native/filament/bridge"
    )
    cmake = (bridge_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    facade = (bridge_dir / "filament_bridge.cpp").read_text(encoding="utf-8")
    header = (bridge_dir / "filament_bridge.h").read_text(encoding="utf-8")

    assert not (bridge_dir / "bridge_screen.cpp").exists()
    assert not (bridge_dir / "bridge_screen.h").exists()
    assert "bridge_screen.cpp" not in cmake
    assert "filament_bridge_create_screen" not in facade
    assert "filament_bridge_create_screen" not in header
    assert "filament_bridge_set_screen_image" not in facade
    assert "filament_bridge_set_screen_image" not in header
    assert "filament_bridge_depth_output_abi_available" in facade
    depth_capability = facade[facade.index("filament_bridge_depth_output_abi_available"):]
    assert "return 1;" in depth_capability[:depth_capability.index("int filament_bridge_get_depth_attachment")]
    assert "filament_bridge_create_eye_swapchain_with_depth" in facade
    assert "bundle.depth = swapchain->depth" in (
        bridge_dir / "bridge_internal.h"
    ).read_text(encoding="utf-8")

def test_legacy_filament_screen_texture_light_path_stays_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/xr_viewer/core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    facade = (root / "native/filament/bridge/filament_bridge.cpp").read_text(
        encoding="utf-8"
    )
    assert "def _update_filament_screen_light" not in source
    assert "filament_bridge_set_screen_light" not in facade
    assert "filament_bridge_set_controller_screen_light" in facade


def test_artemis_controller_lighting_matches_legacy_head_light() -> None:
    root = Path(__file__).resolve().parents[1]
    profile_path = root / "src/xr_viewer/environments/3D_Artemis/profile.json"
    if not profile_path.is_file():
        profile_path = root / "src/xr_viewer/environments/Artemis/profile.json"
    profile = json.loads(
        profile_path.read_text(encoding="utf-8")
    )
    assert profile["env_head_light_color"] == [0.45, 0.45, 0.48]
    assert profile["env_ambient_color"] == [0.06, 0.05, 0.05]
    assert profile["preview_exposure"] == 0.0
    config = (root / "src/xr_viewer/core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    assert "filament_fill_light_intensity: float = 1.0" in config
    assert 'profile.get("env_head_light_color"' in config
    assert '"env_ambient_color", self._filament_ambient_light_color' in config
    assert "bridge.set_ambient_light(self._controller_ambient_light_color())" in config

    native_lighting = (root / "native/filament/bridge/bridge_material.cpp").read_text(
        encoding="utf-8"
    )
    assert "kControllerHeadLightWeight" not in native_lighting
    assert "kControllerTopLightWeight" not in native_lighting
    assert "kLegacyControllerCandelaScale" not in native_lighting
    assert "config->head_light_intensity_candela" in native_lighting
    assert "config->top_light_intensity_candela" in native_lighting
    assert "bridge_material_set_controller_screen_light" in native_lighting
    assert ".lightChannel(0, false).lightChannel(1, true)" in native_lighting
    material_header = (root / "native/filament/bridge/bridge_material.h").read_text(
        encoding="utf-8"
    )
    assert "struct FilamentBridgeLightingConfig;" in material_header

    common = json.loads(
        (root / "src/xr_viewer/environments/common.json").read_text(encoding="utf-8")
    )["filament"]
    assert common["controller_head_light_weight"] == pytest.approx(0.70)
    assert common["controller_top_light_weight"] == pytest.approx(1.0)
    assert common["controller_head_light_offset"] == [0.0, 0.05, 0.0]
    assert common["controller_top_light_offset"] == [0.0, 0.45, -0.18]
    assert common["controller_screen_light_enabled"] is True
    assert common["controller_screen_light_intensity_lux"] == pytest.approx(500.0)
    assert common["controller_screen_light_sample_hz"] == pytest.approx(12.0)
    assert common["glow_sample_hz"] == pytest.approx(30.0)
    assert common["glow_smoothing_seconds"] == pytest.approx(0.10)


@pytest.mark.parametrize("brand", ("HP", "INDEX", "PICO", "QUEST", "VIVE", "YVR"))
@pytest.mark.parametrize("hand", ("left", "right"))
def test_packaged_controller_glb_animation_triplets_have_native_fallbacks(
    brand: str, hand: str
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/xr_viewer/controllers"
        / brand
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
    value_names = {
        name
        for name in names
        if name.endswith("_value")
        and all(
            name.removesuffix("_value") + suffix in names
            for suffix in ("_min", "_max")
        )
    }

    assert value_names
    assert any("trigger_pressed_value" in name for name in value_names)
    assert any("squeeze_pressed_value" in name for name in value_names)
    assert any("thumbstick_pressed_value" in name for name in value_names)
    bridge_source = (
        Path(__file__).resolve().parents[1]
        / "native/filament/bridge/bridge_controller.cpp"
    ).read_text(encoding="utf-8")
    assert all(f'"{value_name}"' in bridge_source for value_name in value_names)


def test_native_controller_animation_preserves_touch_semantics_without_abi_growth() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/filament/bridge/bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    internal = (root / "native/filament/bridge/bridge_internal.h").read_text(
        encoding="utf-8"
    )
    public_header = (root / "native/filament/bridge/filament_bridge.h").read_text(
        encoding="utf-8"
    )

    assert '"joystick_x_touched"' in source
    assert '"joystick_y_touched"' in source
    assert '"joystick_touched"' in source
    assert "controller.button_values[6] * controller.joystick_x" in source
    assert "controller.button_values[6] * controller.joystick_y" in source
    assert "std::array<float, 7> button_values{};" in internal
    assert "uint32_t button_mask" in public_header


def test_native_foreground_uses_one_view_for_multiview_stereo() -> None:
    root = Path(__file__).resolve().parents[1]
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )

    assert "bridge_screen_prepare_frame" not in eye_source
    assert "bridge_screen_bind_stereo_textures" not in eye_source
    assert "bridge->renderer->render(bridge->view);" in eye_source


def test_native_controller_overlay_preserves_composer_color() -> None:
    root = Path(__file__).resolve().parents[1]
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )
    overlay = eye_source[eye_source.index("int bridge_eye_render_controller_overlay"):]
    overlay = overlay[:overlay.index("namespace {")]

    assert "overlay_options.clear = false;" in overlay
    assert "overlay_options.discard = false;" in overlay
    assert "render(eye.controller_view)" in overlay
    assert "render(eye.controller_guide_view)" in overlay
    assert "render(bridge->view)" not in overlay
    assert "render(eye.foreground_view)" not in overlay


def test_native_screen_has_opt_in_multiview_eye_diagnostic() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "native/filament/bridge/bridge_screen.cpp").exists()
