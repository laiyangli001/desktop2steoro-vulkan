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
    import glfw
    import win32con
    import win32gui

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    WDA_EXCLUDEFROMCAPTURE = 0x00000011

    def hide_window_from_capture(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        print("StereoWindow is now hidden from screen capture.")

    def show_window_in_capture(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity(hwnd, 0)
        print("StereoWindow is now visible to screen capture.")

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
