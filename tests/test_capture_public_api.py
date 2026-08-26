import importlib.util
import sys
import types
from pathlib import Path

from path_config import APP_ROOT


def test_capture_public_api_imports():
    from capture import (  # noqa: PLC0415
        CaptureConfig,
        DesktopGrabber,
        capture_frame_to_rgb,
        create_capture_runner,
        create_capture_source,
        prepare_rgb_for_depth_runtime,
    )

    assert CaptureConfig is not None
    assert DesktopGrabber is not None
    assert capture_frame_to_rgb is not None
    assert create_capture_runner is not None
    assert create_capture_source is not None
    assert prepare_rgb_for_depth_runtime is not None


def test_capture_select_loads_without_importing_capture_package():
    path = APP_ROOT / "capture" / "capture_select.py"
    spec = importlib.util.spec_from_file_location("_test_capture_select", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.resolve_capture_tool("DXCamera", os_name="Windows") == "DXCamera"
    assert module.resolve_capture_tool("none", os_name="Darwin") == "ScreenCaptureKit"


def test_capture_select_prefers_desktop_duplication_without_cuda(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_directml", None)

    path = APP_ROOT / "capture" / "capture_select.py"
    spec = importlib.util.spec_from_file_location("_test_capture_select_no_cuda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.resolve_capture_tool("none", os_name="Windows") == "WindowsCapture"
