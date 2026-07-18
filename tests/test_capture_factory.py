import sys
import types

from capture import CaptureConfig, create_capture_runner
from capture.factory import get_desktop_grabber_class
from capture.runners import PollingCaptureRunner


class FakeGrabber:
    pass


def _install_fake_backend(monkeypatch, module_name):
    module = types.ModuleType(module_name)
    module.DesktopGrabber = FakeGrabber
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_windows_dxcamera_selects_dxcamera_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.windows_dxcamera")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Windows", capture_tool="DXCamera"))

    assert cls is FakeGrabber


def test_windows_capture_tools_select_event_runner():
    for capture_tool in ["WindowsCapture", "WindowsCaptureCUDA", "WindowsCaptureROCm"]:
        runner = create_capture_runner(CaptureConfig(os_name="Windows", capture_tool=capture_tool))

        assert runner.__class__.__name__ == "WindowsCaptureEventRunner"


def test_macos_screencapturekit_selects_sck_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.macos_screencapturekit")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Darwin", capture_tool="ScreenCaptureKit"))

    assert cls is FakeGrabber


def test_macos_non_sck_selects_coregraphics_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.macos_coregraphics")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Darwin", capture_tool="CoreGraphics"))

    assert cls is FakeGrabber


def test_linux_selects_mss_backend(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.linux_mss")

    cls = get_desktop_grabber_class(CaptureConfig(os_name="Linux", capture_tool="DXCamera"))

    assert cls is FakeGrabber


def test_polling_tools_create_polling_runner(monkeypatch):
    _install_fake_backend(monkeypatch, "capture.backends.windows_dxcamera")

    for capture_tool in ["DXCamera"]:
        runner = create_capture_runner(CaptureConfig(os_name="Windows", capture_tool=capture_tool))

        assert isinstance(runner, PollingCaptureRunner)
