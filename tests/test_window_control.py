import pytest

from viewer import window_control


pytestmark = pytest.mark.skipif(
    window_control.OS_NAME != "Windows",
    reason="Windows display-affinity API",
)


def test_hide_window_from_capture_uses_full_native_hwnd(monkeypatch):
    hwnd = 0x123456789ABC
    calls = []

    monkeypatch.setattr(
        window_control.glfw,
        "get_win32_window",
        lambda _window: hwnd,
    )
    monkeypatch.setattr(
        window_control,
        "SetWindowDisplayAffinity",
        lambda native_hwnd, affinity: calls.append(
            (native_hwnd, affinity)
        )
        or True,
    )

    assert window_control.hide_window_from_capture(object()) is True
    assert calls == [
        (hwnd, window_control.WDA_EXCLUDEFROMCAPTURE),
    ]


def test_capture_affinity_api_declares_windows_types():
    from ctypes import wintypes

    assert window_control.SetWindowDisplayAffinity.argtypes == [
        wintypes.HWND,
        wintypes.DWORD,
    ]
    assert window_control.SetWindowDisplayAffinity.restype is wintypes.BOOL
