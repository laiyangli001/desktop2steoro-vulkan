from __future__ import annotations

import threading

import numpy as np

from capture.runners import PollingCaptureRunner
from capture.types import CaptureConfig, CapturedFrame, FrameCopyMode


class FakePollingSource:
    def __init__(self):
        self.stopped = False

    def grab(self):
        return np.zeros((4, 8, 3), dtype=np.uint8), (8, 4)

    def stop(self):
        self.stopped = True


def test_polling_capture_runner_marks_explicit_non_zero_copy_metadata():
    source = FakePollingSource()
    shutdown = threading.Event()
    frames = []
    runner = PollingCaptureRunner(
        CaptureConfig(
            output_resolution=(8, 4),
            capture_tool="DXCamera",
            capture_mode="Monitor",
            monitor_index=1,
            os_name="Windows",
        ),
        lambda: source,
    )

    def on_frame(frame):
        frames.append(frame)
        shutdown.set()

    runner.run(shutdown_event=shutdown, on_frame=on_frame)

    assert len(frames) == 1
    captured = frames[0]
    assert isinstance(captured, CapturedFrame)
    assert captured.copy_mode is FrameCopyMode.COPY
    assert captured.metadata["backend"] == "FakePollingSource"
    assert captured.metadata["zero_copy"] is False
    assert captured.capture_tool == "DXCamera"
    assert captured.capture_size == (8, 4)
    assert captured.frame_raw_device == "cpu"
    assert captured.frame_raw_dtype == "uint8"
