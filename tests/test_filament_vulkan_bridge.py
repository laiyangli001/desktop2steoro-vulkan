from __future__ import annotations

import ctypes
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
    assert default_bridge_path().name.startswith("filament_bridge")


def test_pointer_value_accepts_integer_and_c_void_p() -> None:
    assert _as_pointer_value(17) == 17
    assert _as_pointer_value(ctypes.c_void_p(23)) == 23


def test_missing_bridge_library_is_reported() -> None:
    with pytest.raises(FilamentBridgeError, match="unable to load"):
        FilamentVulkanBridge("missing-filament-bridge.dll")


def test_native_bridge_keeps_eye_renderer_and_screen_lifetimes_explicit() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "native/filament/bridge/filament_bridge.cpp").read_text(
                  encoding="utf-8"
              )

    assert "filament::Renderer* renderer = nullptr;" in source
    assert "eye.renderer = bridge->engine->createRenderer();" in source
    assert "bridge->engine->destroy(eye.renderer);" in source
    assert "bool screen_in_scene = false;" in source
    assert "The sampler is required by the material" in source
    assert "diagnostic_frame_count < 8" in source
    assert "[FilamentBridge] acquired eye=" in source
