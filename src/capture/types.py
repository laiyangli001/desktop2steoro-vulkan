from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, TypeAlias


OutputResolution: TypeAlias = int | tuple[int, int]


@dataclass(frozen=True)
class CaptureConfig:
    output_resolution: OutputResolution = 1080
    fps: int = 60
    window_title: str | None = None
    capture_mode: str = "Monitor"
    monitor_index: int = 1
    capture_tool: str | None = None
    os_name: str | None = None


class FrameCopyMode(Enum):
    NONE = "none"
    CLONE = "clone"
    COPY = "copy"
    CPU_NUMPY = "cpu_numpy"
    GPU_TENSOR = "gpu_tensor"


@dataclass(frozen=True)
class CapturedFrame:
    frame: Any
    target_height: OutputResolution
    timestamp: float
    capture_tool: str = ""
    capture_mode: str = ""
    monitor_index: int = 0
    window_title: str = ""
    capture_size: tuple[int, int] | None = None
    frame_raw_type: str = ""
    frame_raw_device: str = ""
    frame_raw_dtype: str = ""
    copy_mode: FrameCopyMode = FrameCopyMode.COPY
    original_format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _frame_raw_type(frame: Any) -> str:
    frame_type = type(frame)
    module = getattr(frame_type, "__module__", "")
    name = getattr(frame_type, "__qualname__", getattr(frame_type, "__name__", ""))
    return f"{module}.{name}" if module and module != "builtins" else str(name)


def _frame_raw_device(frame: Any) -> str:
    device = getattr(frame, "device", "")
    if callable(device):
        try:
            device = device()
        except Exception:
            device = ""
    return str(device) if device is not None else ""


def _frame_raw_dtype(frame: Any) -> str:
    dtype = getattr(frame, "dtype", "")
    return str(dtype) if dtype is not None else ""


def _capture_size(frame: Any) -> tuple[int, int] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        return int(shape[3]), int(shape[2])
    if len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if width is not None and height is not None:
        return int(width), int(height)
    return None


def capture_frame_from_raw(
    frame: Any,
    target_height: OutputResolution,
    timestamp: float,
    *,
    config: CaptureConfig | None = None,
    copy_mode: FrameCopyMode = FrameCopyMode.COPY,
    original_format: str = "",
    metadata: dict[str, Any] | None = None,
    capture_size: tuple[int, int] | None = None,
    frame_raw_type: str | None = None,
    frame_raw_device: str | None = None,
    frame_raw_dtype: str | None = None,
) -> CapturedFrame:
    return CapturedFrame(
        frame=frame,
        target_height=target_height,
        timestamp=timestamp,
        capture_tool=str(config.capture_tool or "") if config is not None else "",
        capture_mode=str(config.capture_mode or "") if config is not None else "",
        monitor_index=int(config.monitor_index) if config is not None else 0,
        window_title=str(config.window_title or "") if config is not None else "",
        capture_size=capture_size if capture_size is not None else _capture_size(frame),
        frame_raw_type=frame_raw_type if frame_raw_type is not None else _frame_raw_type(frame),
        frame_raw_device=frame_raw_device if frame_raw_device is not None else _frame_raw_device(frame),
        frame_raw_dtype=frame_raw_dtype if frame_raw_dtype is not None else _frame_raw_dtype(frame),
        copy_mode=copy_mode,
        original_format=original_format,
        metadata=dict(metadata or {}),
    )


def ensure_captured_frame(
    item: CapturedFrame | tuple[Any, OutputResolution, float],
    *,
    config: CaptureConfig | None = None,
) -> CapturedFrame:
    if isinstance(item, CapturedFrame):
        return item
    frame, target_height, timestamp = item
    return capture_frame_from_raw(frame, target_height, timestamp, config=config)


class CaptureSource(Protocol):
    def grab(self): ...
    def stop(self) -> None: ...


FrameCallback = Callable[[CapturedFrame], None]
ErrorCallback = Callable[[BaseException], None]
StateCallback = Callable[[Any | None, Any | None], None]
Predicate = Callable[[], bool]
PausedCallback = Callable[[str], None]


class CaptureRunner(Protocol):
    def run(
        self,
        *,
        shutdown_event: Any,
        on_frame: FrameCallback,
        on_error: ErrorCallback | None = None,
        on_closed: Callable[[], None] | None = None,
        is_paused: Predicate | None = None,
        is_hard_idle: Predicate | None = None,
        on_paused: PausedCallback | None = None,
        on_session_update: StateCallback | None = None,
        on_tick: Callable[[], None] | None = None,
    ) -> None: ...
