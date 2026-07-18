from __future__ import annotations

from .runners import PollingCaptureRunner
from .types import CaptureConfig

_WINDOWS_EVENT_TOOLS = {"WindowsCapture", "WindowsCaptureROCm", "WindowsCaptureCUDA"}


def _default_os_name():
    from utils import OS_NAME
    return OS_NAME


def _default_capture_tool():
    from utils import CAPTURE_TOOL
    return CAPTURE_TOOL


def normalize_config(config: CaptureConfig | None = None) -> CaptureConfig:
    if config is None:
        return CaptureConfig(os_name=_default_os_name(), capture_tool=_default_capture_tool())
    os_name = config.os_name or _default_os_name()
    capture_tool = config.capture_tool or _default_capture_tool()
    return CaptureConfig(
        output_resolution=config.output_resolution,
        fps=config.fps,
        window_title=config.window_title,
        capture_mode=config.capture_mode,
        monitor_index=config.monitor_index,
        capture_tool=capture_tool,
        os_name=os_name,
    )


def get_desktop_grabber_class(config: CaptureConfig | None = None):
    config = normalize_config(config)
    if config.os_name == "Windows":
        if config.capture_tool in _WINDOWS_EVENT_TOOLS:
            raise RuntimeError(
                f"{config.capture_tool} is an event capture backend; use create_capture_runner instead"
            )
        from .backends.windows_dxcamera import DesktopGrabber
        return DesktopGrabber
    if config.os_name == "Darwin":
        if config.capture_tool == "ScreenCaptureKit":
            from .backends.macos_screencapturekit import DesktopGrabber
            return DesktopGrabber
        from .backends.macos_coregraphics import DesktopGrabber
        return DesktopGrabber
    if config.os_name and config.os_name.startswith("Linux"):
        from .backends.linux_mss import DesktopGrabber
        return DesktopGrabber
    from .backends.windows_dxcamera import DesktopGrabber
    return DesktopGrabber


class DesktopGrabber:
    def __new__(cls, *args, **kwargs):
        backend_cls = get_desktop_grabber_class()
        return backend_cls(*args, **kwargs)


def create_capture_source(config: CaptureConfig | None = None):
    config = normalize_config(config)
    source_cls = get_desktop_grabber_class(config)
    kwargs = {
        "output_resolution": config.output_resolution,
        "fps": config.fps,
        "window_title": config.window_title,
        "capture_mode": config.capture_mode,
        "monitor_index": config.monitor_index,
    }
    return source_cls(**kwargs)


def create_capture_runner(config: CaptureConfig | None = None):
    config = normalize_config(config)
    if config.os_name == "Windows" and config.capture_tool in _WINDOWS_EVENT_TOOLS:
        from .backends.windows_capture_event import WindowsCaptureEventRunner
        return WindowsCaptureEventRunner(config)
    return PollingCaptureRunner(config, lambda: create_capture_source(config))
