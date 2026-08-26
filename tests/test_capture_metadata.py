from __future__ import annotations

from capture.backends.windows_capture_event import _borrow_native_resource
from capture.types import (
    CaptureConfig,
    CapturedFrame,
    FrameCopyMode,
    capture_frame_from_raw,
    ensure_captured_frame,
    capture_frame_from_native_texture,
)


class FakeNativeTexture:
    resource_kind = "d3d11_texture"
    format = "BGRA8"
    width = 1920
    height = 1080
    adapter_luid = 123


class FakeFrame:
    shape = (720, 1280, 4)
    dtype = "uint8"
    device = "cuda:0"


class FakeWgcFrameBuffer:
    def __init__(self):
        self.d3d11_texture = FakeNativeTexture()


def test_windows_capture_borrows_optional_d3d11_resource():
    frame_buffer = FakeWgcFrameBuffer()
    assert _borrow_native_resource(frame_buffer) is frame_buffer.d3d11_texture


def test_capture_frame_from_raw_populates_metadata_contract():
    config = CaptureConfig(
        output_resolution=(3840, 2160),
        capture_tool="WindowsCaptureCUDA",
        capture_mode="Window",
        monitor_index=2,
        window_title="Stereo Viewer",
    )
    metadata = {"backend": "fake"}

    captured = capture_frame_from_raw(
        FakeFrame(),
        (3840, 2160),
        42.0,
        config=config,
        copy_mode=FrameCopyMode.GPU_TENSOR,
        original_format="BGRA",
        metadata=metadata,
    )

    assert isinstance(captured, CapturedFrame)
    assert captured.target_height == (3840, 2160)
    assert captured.timestamp == 42.0
    assert captured.capture_tool == "WindowsCaptureCUDA"
    assert captured.capture_mode == "Window"
    assert captured.monitor_index == 2
    assert captured.window_title == "Stereo Viewer"
    assert captured.capture_size == (1280, 720)
    assert captured.frame_raw_type.endswith("FakeFrame")
    assert captured.frame_raw_device == "cuda:0"
    assert captured.frame_raw_dtype == "uint8"
    assert captured.copy_mode is FrameCopyMode.GPU_TENSOR
    assert captured.original_format == "BGRA"
    assert captured.metadata == {"backend": "fake"}
    assert captured.metadata is not metadata


def test_native_texture_frame_contract_preserves_gpu_resource_metadata():
    captured = capture_frame_from_native_texture(
        FakeNativeTexture(),
        1080,
        3.0,
        config=CaptureConfig(capture_tool="DesktopDuplication"),
    )

    assert captured.copy_mode is FrameCopyMode.NONE
    assert captured.capture_size == (1920, 1080)
    assert captured.frame_raw_device == "d3d11"
    assert captured.metadata["resource_kind"] == "d3d11_texture"
    assert captured.metadata["gpu_to_cpu"] is False
    assert captured.metadata["gpu_copy_count"] == 0
    assert captured.metadata["zero_copy"] is False
    assert captured.metadata["adapter_luid"] == 123


def test_ensure_captured_frame_keeps_new_contract_and_wraps_legacy_tuple():
    captured = CapturedFrame("frame", 1080, 1.0, copy_mode=FrameCopyMode.NONE)

    assert ensure_captured_frame(captured) is captured

    wrapped = ensure_captured_frame(("legacy", 720, 2.0))

    assert wrapped.frame == "legacy"
    assert wrapped.target_height == 720
    assert wrapped.timestamp == 2.0
    assert wrapped.copy_mode is FrameCopyMode.COPY
