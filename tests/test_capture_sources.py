from __future__ import annotations

from gui import capture_sources


def test_gui_default_keeps_cuda_capture(monkeypatch):
    monkeypatch.setattr(capture_sources.devices_module, "DEVICES", {0: {"name": "CUDA 0: RTX"}})
    monkeypatch.setattr(capture_sources.devices_module, "IS_ROCM", False)

    assert capture_sources.get_default_windows_capture_tool() == "WindowsCaptureCUDA"


def test_gui_default_uses_desktop_duplication_without_cuda(monkeypatch):
    monkeypatch.setattr(capture_sources.devices_module, "DEVICES", {0: {"name": "CPU"}})
    monkeypatch.setattr(capture_sources.devices_module, "IS_ROCM", False)

    assert capture_sources.get_default_windows_capture_tool() == "DesktopDuplication"
