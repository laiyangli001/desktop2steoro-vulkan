from __future__ import annotations

from capture.types import (
    CaptureConfig,
    CapturedFrame,
    FrameCopyMode,
    capture_frame_from_raw,
    ensure_captured_frame,
)


class FakeFrame:
    shape = (720, 1280, 4)
    dtype = "uint8"
    device = "cuda:0"


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


def test_ensure_captured_frame_keeps_new_contract_and_wraps_legacy_tuple():
    captured = CapturedFrame("frame", 1080, 1.0, copy_mode=FrameCopyMode.NONE)

    assert ensure_captured_frame(captured) is captured

    wrapped = ensure_captured_frame(("legacy", 720, 2.0))

    assert wrapped.frame == "legacy"
    assert wrapped.target_height == 720
    assert wrapped.timestamp == 2.0
    assert wrapped.copy_mode is FrameCopyMode.COPY
