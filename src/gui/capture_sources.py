import logging
import subprocess

from utils import OS_NAME

from . import devices as devices_module


PRIMARY_MONITOR_SUFFIX = " [Main]"
logger = logging.getLogger(__name__)


def get_default_windows_capture_tool():
    device_name = devices_module.DEVICES.get(0, {}).get("name", "")
    if "CUDA" in device_name and not devices_module.IS_ROCM:
        return "WindowsCaptureCUDA"
    if "CUDA" in device_name and devices_module.IS_ROCM:
        return "WindowsCaptureROCm"
    return "DXCamera"


def get_primary_monitor_index():
    if OS_NAME != "Windows":
        return _get_primary_monitor_index_unix()
    try:
        import win32api, win32con
        primary_monitor_handle = win32api.MonitorFromPoint((0, 0), win32con.MONITOR_DEFAULTTOPRIMARY)
        primary_monitor_info = win32api.GetMonitorInfo(primary_monitor_handle)
        primary_rect = primary_monitor_info["Monitor"]
        primary_left, primary_top, _, _ = primary_rect
        import mss
        with mss.mss() as sct:
            for idx, monitor in enumerate(sct.monitors[1:], start=1):
                if monitor["left"] == primary_left and monitor["top"] == primary_top:
                    return idx
        return 1
    except Exception:
        return 1




def list_monitors():
    """Return display monitors with capture index and user-facing display number."""
    try:
        import mss
        with mss.mss() as sct:
            monitors = [
                {
                    "capture_index": idx,
                    "display_number": idx,
                    "left": mon["left"],
                    "top": mon["top"],
                    "width": mon["width"],
                    "height": mon["height"],
                }
                for idx, mon in enumerate(sct.monitors[1:], start=1)
            ]
    except Exception:
        return []

    if OS_NAME == "Windows":
        _apply_windows_display_numbers(monitors)
        monitors = sorted(monitors, key=lambda mon: mon["display_number"])
        for display_number, mon in enumerate(monitors, start=1):
            mon["device_display_number"] = mon["display_number"]
            mon["display_number"] = display_number
        return monitors

    return sorted(monitors, key=lambda mon: mon["display_number"])


def _apply_windows_display_numbers(monitors):
    rect_to_number = {}
    try:
        import win32api
        for handle, _hdc, rect in win32api.EnumDisplayMonitors():
            info = win32api.GetMonitorInfo(handle)
            display_number = _display_number_from_device(info.get("Device", ""))
            if display_number is not None:
                rect_to_number[tuple(rect)] = display_number
    except Exception:
        return

    for mon in monitors:
        rect = (
            mon["left"],
            mon["top"],
            mon["left"] + mon["width"],
            mon["top"] + mon["height"],
        )
        display_number = rect_to_number.get(rect)
        if display_number is not None:
            mon["display_number"] = display_number


def _display_number_from_device(device_name):
    suffix = str(device_name).upper().rsplit("DISPLAY", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)
def _get_primary_monitor_index_unix():
    try:
        import mss
        with mss.mss() as sct:
            for idx, monitor in enumerate(sct.monitors[1:], start=1):
                if monitor["left"] == 0 and monitor["top"] == 0:
                    return idx
    except Exception:
        pass
    return 1


def list_windows():
    """Return list of {title, handle, rect} for visible windows."""
    if OS_NAME == "Windows":
        return _list_windows_win()
    if OS_NAME == "Darwin":
        return _list_windows_mac()
    return _list_windows_linux()


def _list_windows_win():
    import win32gui
    windows = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                client_rect = win32gui.GetClientRect(hwnd)
                left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
                windows.append({
                    "title": title,
                    "handle": hwnd,
                    "rect": (left, top, client_rect[2], client_rect[3]),
                })
        return True

    win32gui.EnumWindows(callback, None)
    return windows


def _list_windows_mac():
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    windows = []
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    window_info = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
    blacklist = [
        "Window Server", "ControlCenter", "NotificationCenter", "Spotlight",
        "Dock", "FocusModes", "WiFi", "Sound", "UserSwitcher", "Clock",
        "BentoBox", "Bluetooth", "popdown", "AudioVideoModule",
        "ScreenMirroring", "SystemUIServer", "CoreServicesUIAgent",
        "TextInputMenuAgent", "com.apple.controlcenter", "loginwindow",
    ]
    for win in window_info:
        title = win.get("kCGWindowName", "") or ""
        owner = win.get("kCGWindowOwnerName", "")
        layer = win.get("kCGWindowLayer", 0)
        bounds = win.get("kCGWindowBounds", {})
        if not title.strip():
            continue
        if owner in blacklist or title in blacklist:
            continue
        if layer >= 1000:
            continue
        if win.get("kCGWindowAlpha", 1.0) == 0.0:
            continue
        if title.strip().lower().startswith(("item-", "window-")):
            continue
        if "X" in bounds and "Y" in bounds and "Width" in bounds and "Height" in bounds:
            w, h = bounds["Width"], bounds["Height"]
            if w < 10 or h < 10:
                continue
            windows.append({
                "title": title.strip(),
                "handle": win["kCGWindowNumber"],
                "rect": (bounds["X"], bounds["Y"], w, h),
            })
    return windows


def _list_windows_linux():
    windows = []
    try:
        result = subprocess.check_output(["wmctrl", "-lG"], timeout=2).decode("utf-8").splitlines()
        for line in result:
            parts = line.split(None, 7)
            if len(parts) >= 8:
                _, _, x_str, y_str, w_str, h_str, _, title = parts
                try:
                    x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)
                    if title.strip():
                        windows.append({"title": title.strip(), "handle": None, "rect": (x, y, w, h)})
                except ValueError:
                    continue
    except Exception as e:
        logger.warning("Linux window enumeration failed (install wmctrl): %s", e)
    return windows


def get_monitor_index_for_point(x, y):
    try:
        import mss
        with mss.mss() as sct:
            for idx, mon in enumerate(sct.monitors[1:], start=1):
                left, top = mon["left"], mon["top"]
                right, bottom = left + mon["width"], top + mon["height"]
                if left <= x < right and top <= y < bottom:
                    return idx
    except Exception:
        pass
    return get_primary_monitor_index()


def get_capture_tool_options(device_label):
    if OS_NAME == "Darwin":
        return ["ScreenCaptureKit", "Quartz"]
    if OS_NAME != "Windows":
        return ["DXCamera"]
    device_label_upper = device_label.upper()
    is_nvidia = "CUDA" in device_label_upper and not devices_module.IS_ROCM
    if is_nvidia:
        return ["WindowsCaptureCUDA", "WindowsCapture", "DXCamera"]
    if "CUDA" in device_label_upper and devices_module.IS_ROCM:
        return ["WindowsCaptureROCm", "WindowsCapture", "DXCamera"]
    return ["DXCamera", "WindowsCapture"]
