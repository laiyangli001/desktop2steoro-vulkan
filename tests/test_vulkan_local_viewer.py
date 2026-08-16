from __future__ import annotations

import numpy as np

from viewer.vulkan_local_viewer import (
    LOCAL_VIEWER_SOURCE_FORMAT,
    choose_srgb_surface_format,
    direct_display_capability,
    fit_rect,
    frame_to_cuda_rgba,
    frame_to_rgba_bytes,
    glfw_monitor_array_index,
    is_exclusive_fullscreen_toggle,
    present_fps_if_due,
    should_restore_persistent_fullscreen,
    VulkanLocalViewer,
    VulkanLocalViewerConfig,
)


class _Format:
    def __init__(self, value: int) -> None:
        self.format = value


class _VulkanFormats:
    VK_FORMAT_B8G8R8A8_SRGB = 50
    VK_FORMAT_R8G8B8A8_SRGB = 43


class _GlfwKeys:
    KEY_ENTER = 257
    PRESS = 1
    RELEASE = 0
    MOD_ALT = 4


def test_fit_rect_letterboxes_wide_sbs_frame() -> None:
    assert fit_rect((3840, 1080), (1280, 720)) == (0, 180, 1280, 360)


def test_frame_to_rgba_bytes_converts_chw_float() -> None:
    frame = np.zeros((3, 2, 5), dtype=np.float32)
    frame[0] = 1.0
    packed, width, height = frame_to_rgba_bytes(frame)
    image = np.frombuffer(packed, dtype=np.uint8).reshape(height, width, 4)
    assert (width, height) == (5, 2)
    assert image[0, 0].tolist() == [255, 0, 0, 255]


def test_frame_to_rgba_bytes_preserves_hwc_uint8() -> None:
    frame = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    packed, width, height = frame_to_rgba_bytes(frame)
    assert (width, height) == (1, 1)
    assert list(packed) == [1, 2, 3, 4]


def test_cuda_packer_does_not_take_a_cpu_frame_down_the_interop_path() -> None:
    assert frame_to_cuda_rgba(np.zeros((2, 2, 3), dtype=np.uint8)) is None


def test_local_viewer_uses_srgb_source_and_preferred_surface_format() -> None:
    selected, is_srgb = choose_srgb_surface_format(
        [_Format(99), _Format(_VulkanFormats.VK_FORMAT_R8G8B8A8_SRGB)],
        _VulkanFormats,
    )
    assert LOCAL_VIEWER_SOURCE_FORMAT == "VK_FORMAT_R8G8B8A8_SRGB"
    assert selected.format == _VulkanFormats.VK_FORMAT_R8G8B8A8_SRGB
    assert is_srgb


def test_local_viewer_reports_unorm_surface_fallback() -> None:
    selected, is_srgb = choose_srgb_surface_format([_Format(99)], _VulkanFormats)
    assert selected.format == 99
    assert not is_srgb


def test_direct_display_requires_surface_and_acquire_extensions() -> None:
    assert direct_display_capability(
        {"VK_KHR_display", "VK_EXT_direct_mode_display"}
    ) == (True, ())
    supported, missing = direct_display_capability({"VK_KHR_surface"})
    assert not supported
    assert missing == (
        "VK_KHR_display",
        "VK_EXT_direct_mode_display/VK_NV_acquire_winrt_display",
    )


def test_alt_enter_toggles_exclusive_fullscreen_once_per_press() -> None:
    assert is_exclusive_fullscreen_toggle(257, 1, 4, _GlfwKeys)
    assert not is_exclusive_fullscreen_toggle(257, 0, 4, _GlfwKeys)
    assert not is_exclusive_fullscreen_toggle(257, 1, 0, _GlfwKeys)


def test_gui_monitor_number_maps_to_glfw_array_index() -> None:
    assert glfw_monitor_array_index(1, 3) == 0
    assert glfw_monitor_array_index(2, 3) == 1
    assert glfw_monitor_array_index(99, 3) == 2
    assert glfw_monitor_array_index(0, 3) == 0


def test_persistent_fullscreen_restores_hidden_minimized_or_non_topmost_window() -> None:
    assert should_restore_persistent_fullscreen(True, False, False, True)
    assert should_restore_persistent_fullscreen(True, True, True, True)
    assert should_restore_persistent_fullscreen(True, True, False, False)
    assert not should_restore_persistent_fullscreen(True, True, False, True)
    assert not should_restore_persistent_fullscreen(False, False, True, False)


def test_present_fps_is_emitted_only_after_five_seconds() -> None:
    assert present_fps_if_due(299, 4.99) is None
    assert present_fps_if_due(300, 5.0) == 60.0


def test_present_fps_feedback_does_not_depend_on_log_visibility() -> None:
    samples = []
    viewer = VulkanLocalViewer(
        VulkanLocalViewerConfig(
            show_fps=False,
            on_sbs_fps=lambda fps: samples.append(fps) or 65,
        )
    )
    viewer._fps_started = 0.0
    viewer._fps_frames = 299

    assert present_fps_if_due(viewer._fps_frames + 1, 5.0) == 60.0
    assert viewer._report_present_fps(60.0) == 65
    assert samples == [60.0]
