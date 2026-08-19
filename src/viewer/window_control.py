import platform
import time


OS_NAME = platform.system()


if OS_NAME == "Darwin":
    try:
        import Quartz
    except ImportError:
        Quartz = None

    KEY_F = 3
    MODIFY_FLAGS = (
        Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand
        if Quartz is not None
        else 0
    )

    def send_ctrl_cmd_f(key=KEY_F, flags=MODIFY_FLAGS):
        if Quartz is None:
            return
        ev_down = Quartz.CGEventCreateKeyboardEvent(None, key, True)
        Quartz.CGEventSetFlags(ev_down, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)

        time.sleep(0.02)

        ev_up = Quartz.CGEventCreateKeyboardEvent(None, key, False)
        Quartz.CGEventSetFlags(ev_up, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)

else:
    def send_ctrl_cmd_f(*args, **kwargs):
        return None


if OS_NAME == "Windows":
    import ctypes
    from ctypes import wintypes

    import glfw
    import win32con
    import win32gui

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    SetWindowDisplayAffinity.restype = wintypes.BOOL
    WDA_NONE = 0x00000000
    WDA_EXCLUDEFROMCAPTURE = 0x00000011

    def _set_window_capture_affinity(glfw_window, affinity):
        hwnd = glfw.get_win32_window(glfw_window)
        if not hwnd:
            print(
                "[WindowCapture] Failed to resolve the native HWND.",
                flush=True,
            )
            return False
        ctypes.windll.kernel32.SetLastError(0)
        if not bool(SetWindowDisplayAffinity(hwnd, affinity)):
            error_code = int(ctypes.windll.kernel32.GetLastError())
            print(
                "[WindowCapture] SetWindowDisplayAffinity failed: "
                f"hwnd=0x{int(hwnd):X} affinity=0x{int(affinity):X} "
                f"winerror={error_code}",
                flush=True,
            )
            return False
        return True

    def hide_window_from_capture(glfw_window):
        hidden = _set_window_capture_affinity(
            glfw_window,
            WDA_EXCLUDEFROMCAPTURE,
        )
        if hidden:
            print(
                "[WindowCapture] SBS output is excluded from screen capture.",
                flush=True,
            )
        return hidden

    def show_window_in_capture(glfw_window):
        visible = _set_window_capture_affinity(glfw_window, WDA_NONE)
        if visible:
            print(
                "[WindowCapture] SBS output is visible to screen capture.",
                flush=True,
            )
        return visible

    def set_window_to_bottom(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        if hwnd:
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_BOTTOM,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
            )

else:
    def hide_window_from_capture(*args, **kwargs):
        return None

    def show_window_in_capture(*args, **kwargs):
        return None

    def set_window_to_bottom(*args, **kwargs):
        return None
