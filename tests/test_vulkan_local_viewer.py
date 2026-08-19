from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import numpy as np

import viewer.vulkan_local_viewer as local_viewer_module
from viewer.vulkan_local_viewer import (
    LOCAL_VIEWER_SOURCE_FORMAT,
    choose_srgb_surface_format,
    configure_glfw_window_hints,
    direct_display_capability,
    fit_rect,
    frame_to_cuda_rgba,
    frame_to_rgba_bytes,
    depth_preview_frame,
    depth_to_effective_disparity,
    depth_to_red_blue_rgb,
    effective_disparity_to_red_blue_rgb,
    is_exclusive_fullscreen_toggle,
    present_fps_if_due,
    should_restore_persistent_fullscreen,
    run_vulkan_local_viewer,
    VulkanLocalViewer,
    VulkanLocalViewerConfig,
)


class _Format:
    def __init__(self, value: int) -> None:
        self.format = value


class _WindowHintGlfw:
    CLIENT_API = 1
    NO_API = 2
    RESIZABLE = 3
    AUTO_ICONIFY = 4
    VISIBLE = 5
    DECORATED = 6
    FLOATING = 7
    FOCUS_ON_SHOW = 8
    TRUE = 1
    FALSE = 0

    def __init__(self):
        self.hints = {}
        self.reset_count = 0

    def default_window_hints(self):
        self.hints = {}
        self.reset_count += 1

    def window_hint(self, key, value):
        self.hints[key] = value


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


def test_cuda_float_chw_packer_returns_rgba8_without_host_round_trip() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    frame = torch.zeros((1, 3, 4, 6), device="cuda", dtype=torch.float32)
    frame[:, 0] = 1.0
    packed = frame_to_cuda_rgba(frame)
    assert packed is not None
    assert tuple(packed.shape) == (4, 6, 4)
    assert packed.dtype == torch.uint8
    assert packed[0, 0].tolist() == [255, 0, 0, 255]


def test_local_viewer_uses_srgb_source_and_preferred_surface_format() -> None:
    selected, is_srgb = choose_srgb_surface_format(
        [_Format(99), _Format(_VulkanFormats.VK_FORMAT_R8G8B8A8_SRGB)],
        _VulkanFormats,
    )
    assert LOCAL_VIEWER_SOURCE_FORMAT == "VK_FORMAT_R8G8B8A8_SRGB"
    assert selected.format == _VulkanFormats.VK_FORMAT_R8G8B8A8_SRGB
    assert is_srgb


def test_fullscreen_and_debug_preview_reset_independent_window_hints() -> None:
    glfw = _WindowHintGlfw()

    configure_glfw_window_hints(glfw, fullscreen=True)
    assert glfw.hints[glfw.VISIBLE] == glfw.FALSE
    assert glfw.hints[glfw.DECORATED] == glfw.TRUE
    assert glfw.hints[glfw.FLOATING] == glfw.TRUE

    configure_glfw_window_hints(glfw, fullscreen=False)

    assert glfw.reset_count == 2
    assert glfw.hints[glfw.VISIBLE] == glfw.TRUE
    assert glfw.hints[glfw.DECORATED] == glfw.TRUE
    assert glfw.hints[glfw.FLOATING] == glfw.FALSE
    assert glfw.hints[glfw.FOCUS_ON_SHOW] == glfw.TRUE


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
            on_sbs_fps=lambda fps, frames: samples.append((fps, frames)) or 65,
        )
    )
    viewer._fps_started = 0.0
    viewer._fps_frames = 299

    assert present_fps_if_due(viewer._fps_frames + 1, 5.0) == 60.0
    assert viewer._report_present_fps(60.0, 300) == 65
    assert samples == [(60.0, 300)]


def test_local_viewer_reports_queue_and_present_breakdown(monkeypatch) -> None:
    shutdown = threading.Event()
    runtime_q = queue.Queue()
    runtime_q.put((SimpleNamespace(sbs=object()), 0.0))
    runtime_q.put((SimpleNamespace(sbs=object()), 0.0))
    counts = []
    timings = []

    class FakeViewer:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        def present(self, _frame):
            shutdown.set()

        def poll_events(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(local_viewer_module, "VulkanLocalViewer", FakeViewer)
    run_vulkan_local_viewer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        config=VulkanLocalViewerConfig(
            on_breakdown_inc=lambda name, amount: counts.append((name, amount)),
            on_breakdown_add_time=lambda name, seconds: timings.append((name, seconds)),
        ),
    )

    assert counts.count(("viewer_get", 1)) == 2
    assert ("viewer_drop", 1) in counts
    assert ("local_presented_frame", 1) in counts
    assert timings[0][0] == "local_present"
    assert timings[0][1] >= 0.0


def test_window_preview_duplicates_output_without_replacing_fullscreen(monkeypatch) -> None:
    shutdown = threading.Event()
    runtime_q = queue.Queue()
    frame = object()
    depth_frame = object()
    runtime_q.put((SimpleNamespace(sbs=frame), 0.0))
    created = []

    class FakeViewer:
        def __init__(self, config):
            self.config = config
            self.frames = []
            created.append(self)

        def initialize(self):
            pass

        def present(self, value):
            self.frames.append(value)
            if len(created) == 2 and all(item.frames for item in created):
                shutdown.set()

        def poll_events(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(local_viewer_module, "VulkanLocalViewer", FakeViewer)
    monkeypatch.setattr(local_viewer_module, "depth_preview_frame", lambda _result: depth_frame)
    run_vulkan_local_viewer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        config=VulkanLocalViewerConfig(
            monitor_index=3,
            fullscreen=True,
            window_preview=True,
            preview_monitor_index=1,
        ),
    )

    assert len(created) == 2
    output, preview = created
    assert output.config.fullscreen is True
    assert output.config.monitor_index == 3
    assert preview.config.fullscreen is False
    assert preview.config.monitor_index == 1
    assert preview.config.exclude_from_capture is False
    assert output.frames == [frame]
    assert preview.frames == [depth_frame]


def test_depth_preview_uses_one_matched_depth_map_size_not_sbs() -> None:
    torch = __import__("torch")
    depth = torch.tensor([[[[0.0, 1.0], [0.25, 0.75]]]])
    result = SimpleNamespace(
        depth=depth,
        sbs=torch.ones(1, 3, 4, 8),
        output_eye_size=(4, 4),
        debug_info={"depth_strength": 0.25},
    )

    preview = depth_preview_frame(result)

    assert tuple(preview.shape) == (1, 3, 4, 4)
    assert preview[0, :, 0, 0].tolist() == [0.0, 0.0, 1.0]
    assert preview[0, :, 0, -1].tolist() == [1.0, 0.0, 0.0]
    assert float(preview.min()) == 0.0
    assert float(preview.max()) == 1.0


def test_depth_preview_maps_hot_reloaded_depth_strength_blue_to_red() -> None:
    torch = __import__("torch")
    depth = torch.tensor([[[[0.0, 0.5, 1.0]]]])

    low = depth_preview_frame(SimpleNamespace(
        depth=depth,
        output_eye_size=(3, 1),
        debug_info={"depth_strength": 0.0},
    ))
    middle = depth_preview_frame(SimpleNamespace(
        depth=depth,
        output_eye_size=(3, 1),
        debug_info={"depth_strength": 0.25},
    ))
    high = depth_preview_frame(SimpleNamespace(
        depth=depth,
        output_eye_size=(3, 1),
        debug_info={"depth_strength": 0.5},
    ))

    assert low[0].permute(1, 2, 0).tolist() == [[[0.0, 0.0, 1.0]] * 3]
    assert middle[0].permute(1, 2, 0).tolist() == [[
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.5],
        [1.0, 0.0, 0.0],
    ]]
    assert high[0].permute(1, 2, 0).tolist() == [[[1.0, 0.0, 0.0]] * 3]
    assert torch.equal(
        depth_to_effective_disparity(depth, 0.25),
        depth,
    )
    assert effective_disparity_to_red_blue_rgb(torch.full_like(depth, 0.5))[0, :, 0, 0].tolist() == [0.5, 0.0, 0.5]


def test_depth_color_map_transitions_from_far_blue_to_near_red() -> None:
    torch = __import__("torch")
    depth = torch.tensor([[[[0.0, 0.25, 0.5, 0.75, 1.0]]]])

    rgb = depth_to_red_blue_rgb(depth)

    assert rgb[0, :, 0].T.tolist() == [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
