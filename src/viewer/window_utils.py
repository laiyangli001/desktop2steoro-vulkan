from __future__ import annotations

import platform
import subprocess
import time

OS_NAME = platform.system()

# Get window lists
if OS_NAME == "Windows":
    try:
        import win32gui
    except ImportError:
        win32gui = None

    def list_windows():
        windows = []
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((title, hwnd))
            return True

        win32gui.EnumWindows(callback, None)
        return windows
elif OS_NAME == "Darwin":
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except ImportError:
        CGWindowListCopyWindowInfo = None

    def list_windows():
        windows = []
        options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
        window_info = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        # System UI processes we want to ignore
        blacklist = {
            "Window Server",
            "ControlCenter",
            "NotificationCenter",
            "Spotlight",
            "Dock",
            "SystemUIServer",
            "CoreServicesUIAgent",
            "TextInputMenuAgent",
        }
        for win in window_info:
            title = win.get("kCGWindowName", "") or ""
            owner = win.get("kCGWindowOwnerName", "")
            layer = win.get("kCGWindowLayer", 0)
            bounds = win.get("kCGWindowBounds", {})
            # Filtering rules
            if not title.strip():
                continue
            if owner in blacklist:
                continue
            if title.strip().lower().startswith("item-"):
                continue
            if bounds.get("Y", 1) == 0:
                continue
            windows.append((title.strip(), win["kCGWindowNumber"]))
        return windows
else:
    import subprocess
    def list_windows():
        windows = []
        try:
            result = subprocess.check_output(["wmctrl", "-l"]).decode("utf-8").splitlines()
            for line in result:
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    _, _, _, title = parts
                    if title.strip():
                        windows.append((title.strip(), None))
        except Exception as e:
            print("Linux window listing error:", e)
        return windows

def is_window_visible_on_screen(window_title_search, partial_match=True, timeout=2.0):
    """
    Check if a window with the given title is actually visible on screen.
    
    Args:
        window_title_search: Title or partial title to search for
        partial_match: If True, search for windows containing the search string
        timeout: How long to keep trying (seconds)
    
    Returns:
        tuple: (found, window_title, window_id_or_handle)
    """
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            windows = list_windows()
            
            for title, window_id in windows:
                if partial_match:
                    if window_title_search.lower() in title.lower():
                        return True
                elif title == window_title_search:
                    return True
            
            time.sleep(0.1)
        except Exception as e:
            print(f"[Window Check] Error listing windows: {e}")
            time.sleep(0.5)
    
    return False

