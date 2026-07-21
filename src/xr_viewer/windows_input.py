# Desktop2Stereo OpenXR viewer: OS mouse, keyboard, and scroll injection helpers.

import ctypes
import sys

__all__ = [
    '_set_cursor_pos',
    '_send_mouse_flags',
    '_send_key',
    '_get_desktop_size',
    '_send_vscroll',
    '_send_hscroll',
    '_U32',
    '_MOUSEEVENTF_LEFTDOWN',
    '_MOUSEEVENTF_LEFTUP',
    '_MOUSEEVENTF_RIGHTDOWN',
    '_MOUSEEVENTF_RIGHTUP',
]

# Windows input helpers (no-op on non-Windows)

if sys.platform == "win32":
    _U32 = ctypes.windll.user32

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _INPUT(ctypes.Structure):
        class _I(ctypes.Union):
            _fields_ = [("mi", _MOUSEINPUT)]
        _anonymous_ = ("_i",)
        _fields_ = [("type", ctypes.c_ulong), ("_i", _I)]

    _MOUSEEVENTF_MOVE     = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP   = 0x0004
    _MOUSEEVENTF_RIGHTDOWN= 0x0008
    _MOUSEEVENTF_RIGHTUP  = 0x0010
    _MOUSEEVENTF_ABSOLUTE = 0x8000
    _MOUSEEVENTF_WHEEL    = 0x0800
    _MOUSEEVENTF_HWHEEL   = 0x1000
    _KEYEVENTF_KEYUP      = 0x0002

    def _set_cursor_pos(x, y):
        # Use SetCursorPos with virtual-desktop pixel coordinates -works across all
        # monitors.  The old SendInput+MOVE+ABSOLUTE approach required manual
        # normalisation against the primary-monitor size and was fragile for
        # multi-monitor setups where the primary monitor isn't at (0,0).
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def _send_mouse_flags(flags):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = flags
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False):
        kbd = ctypes.windll.user32.keybd_event
        # Press modifiers (chord support: Ctrl+C, Alt+Tab, Win+R, etc.)
        if ctrl:  kbd(0x11, 0, 0, 0)             # VK_CONTROL down
        if shift: kbd(0x10, 0, 0, 0)             # VK_SHIFT down
        if alt:   kbd(0x12, 0, 0, 0)             # VK_MENU (Alt) down
        if win:   kbd(0x5B, 0, 0, 0)             # VK_LWIN down
        kbd(vk, 0, 0, 0)                          # key down
        kbd(vk, 0, _KEYEVENTF_KEYUP, 0)           # key up
        # Release modifiers in reverse
        if win:   kbd(0x5B, 0, _KEYEVENTF_KEYUP, 0)
        if alt:   kbd(0x12, 0, _KEYEVENTF_KEYUP, 0)
        if shift: kbd(0x10, 0, _KEYEVENTF_KEYUP, 0)
        if ctrl:  kbd(0x11, 0, _KEYEVENTF_KEYUP, 0)

    def _get_desktop_size():
        return _U32.GetSystemMetrics(0), _U32.GetSystemMetrics(1)

    def _send_vscroll(amount):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_WHEEL
        inp.mi.mouseData = ctypes.c_ulong(int(amount) & 0xFFFFFFFF)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_hscroll(amount):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_HWHEEL
        inp.mi.mouseData = ctypes.c_ulong(int(amount) & 0xFFFFFFFF)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
else:
    def _set_cursor_pos(x, y): pass
    def _send_mouse_flags(flags): pass
    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False): pass
    def _send_vscroll(amount): pass
    def _send_hscroll(amount): pass
    def _get_desktop_size(): return (1920, 1080)
    _MOUSEEVENTF_LEFTDOWN  = 0x0002
    _MOUSEEVENTF_LEFTUP    = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP   = 0x0010
