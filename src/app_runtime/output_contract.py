"""Unified output contract shared by preview, OpenXR, and encoder sinks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class VulkanStereoOutputFrame:
    """One GPU frame published to every output target."""

    frame_id: int
    timestamp: float
    left_eye: Any
    right_eye: Any
    sbs: Any | None = None
    output_format: str = "openxr_full_synthesis_eyes"
    ready_timeline: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    color_space: str = "srgb"

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("output frame_id must not be negative")
        if float(self.timestamp) < 0:
            raise ValueError("output timestamp must not be negative")
        if self.left_eye is None or self.right_eye is None:
            raise ValueError("output frame requires both left_eye and right_eye")
        if not str(self.output_format).strip():
            raise ValueError("output_format must not be empty")
        if str(self.color_space).strip().lower() not in {"srgb", "linear"}:
            raise ValueError("output color_space must be srgb or linear")
        if self.ready_timeline is not None and int(self.ready_timeline) < 0:
            raise ValueError("ready_timeline must not be negative")


class VulkanStereoOutputSink(Protocol):
    def submit(self, frame: VulkanStereoOutputFrame) -> None: ...


class LatestFrameOutputRouter:
    """Fan out only the newest frame and bound sink-side backlog to one frame."""

    def __init__(self) -> None:
        self._sinks: dict[str, VulkanStereoOutputSink] = {}
        self._latest: VulkanStereoOutputFrame | None = None
        self._closed = False

    def add_sink(self, name: str, sink: VulkanStereoOutputSink) -> None:
        if self._closed:
            raise RuntimeError("output router is closed")
        key = str(name).strip()
        if not key:
            raise ValueError("output sink name must not be empty")
        if key in self._sinks:
            raise ValueError(f"output sink already exists: {key}")
        self._sinks[key] = sink

    def remove_sink(self, name: str) -> None:
        self._sinks.pop(str(name).strip(), None)

    @property
    def latest(self) -> VulkanStereoOutputFrame | None:
        return self._latest

    @property
    def sink_names(self) -> tuple[str, ...]:
        return tuple(self._sinks)

    def publish(self, frame: VulkanStereoOutputFrame) -> None:
        if self._closed:
            raise RuntimeError("output router is closed")
        self._latest = frame
        for sink in tuple(self._sinks.values()):
            sink.submit(frame)

    def close(self) -> None:
        self._sinks.clear()
        self._latest = None
        self._closed = True
