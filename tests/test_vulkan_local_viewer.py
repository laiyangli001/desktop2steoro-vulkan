from __future__ import annotations

import sys

import numpy as np

from viewer.vulkan_local_viewer import (
    LOCAL_VIEWER_SOURCE_FORMAT,
    choose_srgb_surface_format,
    direct_display_capability,
    fit_rect,
    frame_to_cuda_rgba,
    frame_to_rgba_bytes,
    glfw_monitor_array_index,
    glfw_monitor_for_mss_index,
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


def test_key_handler_ignores_release_events() -> None:
    keys = _GlfwKeys
    calls = []

    class _Glfw:
        KEY_ENTER = keys.KEY_ENTER
        KEY_SPACE = 32
        KEY_ESCAPE = 256
        KEY_LEFT = 263
        KEY_RIGHT = 262
        KEY_F = 70
        MOD_ALT = keys.MOD_ALT
        PRESS = keys.PRESS

        @staticmethod
        def set_window_should_close(window, value):
            calls.append(("close", value))

    viewer = VulkanLocalViewer(VulkanLocalViewerConfig())
    viewer.glfw = _Glfw
    viewer.window = object()
    viewer._on_key(None, keys.KEY_ENTER, 0, keys.RELEASE, 0)
    assert calls == []
    assert viewer._exclusive_fullscreen is False


def test_key_handler_escape_closes_window() -> None:
    keys = _GlfwKeys
    calls = []

    class _Glfw:
        KEY_ENTER = keys.KEY_ENTER
        KEY_SPACE = 32
        KEY_ESCAPE = 256
        KEY_LEFT = 263
        KEY_RIGHT = 262
        KEY_F = 70
        MOD_ALT = keys.MOD_ALT
        PRESS = keys.PRESS

        @staticmethod
        def set_window_should_close(window, value):
            calls.append(("close", value))

    viewer = VulkanLocalViewer(VulkanLocalViewerConfig())
    viewer.glfw = _Glfw
    viewer.window = object()
    viewer._on_key(None, 256, 0, keys.PRESS, 0)
    assert calls == [("close", True)]


def test_key_handler_f_toggles_fps_display() -> None:
    keys = _GlfwKeys

    class _Glfw:
        KEY_ENTER = keys.KEY_ENTER
        KEY_SPACE = 32
        KEY_ESCAPE = 256
        KEY_LEFT = 263
        KEY_RIGHT = 262
        KEY_F = 70
        MOD_ALT = keys.MOD_ALT
        PRESS = keys.PRESS

    viewer = VulkanLocalViewer(VulkanLocalViewerConfig(show_fps=False))
    viewer.glfw = _Glfw
    viewer.window = object()
    assert viewer._current_show_fps() is False
    viewer._on_key(None, 70, 0, keys.PRESS, 0)
    assert viewer._current_show_fps() is True
    viewer._on_key(None, 70, 0, keys.PRESS, 0)
    assert viewer._current_show_fps() is False


def test_key_handler_enter_toggles_exclusive_fullscreen() -> None:
    keys = _GlfwKeys

    class _Glfw:
        KEY_ENTER = keys.KEY_ENTER
        KEY_SPACE = 32
        KEY_ESCAPE = 256
        KEY_LEFT = 263
        KEY_RIGHT = 262
        KEY_F = 70
        MOD_ALT = keys.MOD_ALT
        PRESS = keys.PRESS

    class _Monitor:
        pass

    viewer = VulkanLocalViewer(VulkanLocalViewerConfig())
    viewer.glfw = _Glfw
    viewer.window = object()
    viewer._target_monitor = _Monitor()
    viewer._windowed_rect = (10, 20, 800, 600)
    viewer._set_exclusive_fullscreen = lambda enabled: setattr(
        viewer, "_exclusive_fullscreen", enabled
    )
    viewer._on_key(None, keys.KEY_ENTER, 0, keys.PRESS, 0)
    assert viewer._exclusive_fullscreen is True


def test_gui_monitor_number_maps_to_glfw_array_index() -> None:
    assert glfw_monitor_array_index(1, 3) == 0
    assert glfw_monitor_array_index(2, 3) == 1
    assert glfw_monitor_array_index(99, 3) == 2
    assert glfw_monitor_array_index(0, 3) == 0


def test_gui_monitor_number_matches_glfw_monitor_by_geometry(monkeypatch) -> None:
    import types

    class _ModeSize:
        width = 1920
        height = 1080

    class _Mode:
        size = _ModeSize()

    class _Sct:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        monitors = [
            {"left": 0, "top": 0, "width": 3840, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        ]

    class _Mss:
        @staticmethod
        def mss():
            return _Sct()

    monkeypatch.setitem(
        sys.modules, "mss", types.SimpleNamespace(mss=_Mss.mss)
    )

    class _Glfw:
        @staticmethod
        def get_monitor_pos(name):
            return {"a": (0, 0), "b": (1920, 0)}[name]

        @staticmethod
        def get_video_mode(name):
            return _Mode()

    monitors = ["a", "b"]
    assert glfw_monitor_for_mss_index(_Glfw, 1, monitors) == "a"
    assert glfw_monitor_for_mss_index(_Glfw, 2, monitors) == "b"


def test_gui_monitor_number_falls_back_to_primary_when_geometry_missing(
    monkeypatch,
) -> None:
    import types

    class _Sct:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        monitors = [
            {"left": 0, "top": 0, "width": 3840, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        ]

    class _Mss:
        @staticmethod
        def mss():
            return _Sct()

    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(mss=_Mss.mss))

    class _Glfw:
        @staticmethod
        def get_monitor_pos(name):
            return {"a": (0, 0), "b": (0, 2000)}[name]

        @staticmethod
        def get_video_mode(name):
            class _ModeSize:
                width = 800
                height = 600

            class _Mode:
                size = _ModeSize()

            return _Mode()

    assert glfw_monitor_for_mss_index(_Glfw, 2, ["a"]) == "a"
    assert glfw_monitor_for_mss_index(_Glfw, 99, ["a"]) == "a"


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
