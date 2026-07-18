from __future__ import annotations

import json
import os
import subprocess
import threading
import time


DEFAULT_SIGNAL_STATE = {
    "gpu_3d_util": 0.0,
    "gpu_video_decode_util": 0.0,
    "input_activity": 0.0,
    "idle_seconds": 0.0,
    "audio_active": False,
    "maximized": False,
    "foreground_process": "",
    "fullscreen": False,
}


PLAYER_PROCESS_TOKENS = (
    "vlc",
    "mpv",
    "potplayer",
    "player",
    "chrome",
    "edge",
    "firefox",
)


def clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def query_process_name(pid, *, os_name: str) -> str:
    if os_name != "Windows" or not pid:
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return os.path.basename(buffer.value)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return ""
    return ""


def sample_window_input_context(*, os_name: str) -> dict:
    result = {
        "input_activity": 0.0,
        "idle_seconds": 0.0,
        "maximized": False,
        "foreground_process": "",
        "fullscreen": False,
    }
    if os_name != "Windows":
        return result
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            tick = ctypes.windll.kernel32.GetTickCount()
            idle_s = max(0.0, float((tick - info.dwTime) & 0xFFFFFFFF) / 1000.0)
            result["idle_seconds"] = idle_s
            if idle_s < 0.25:
                result["input_activity"] = 1.0
            elif idle_s < 1.0:
                result["input_activity"] = 0.7
            elif idle_s < 3.0:
                result["input_activity"] = 0.35

        try:
            import win32api
            import win32gui
            import win32process

            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                result["foreground_process"] = query_process_name(pid, os_name=os_name)
                result["maximized"] = bool(win32gui.IsZoomed(hwnd))
                rect = win32gui.GetWindowRect(hwnd)
                monitor = win32api.MonitorFromWindow(hwnd, 2)
                monitor_info = win32api.GetMonitorInfo(monitor)
                mx1, my1, mx2, my2 = monitor_info.get("Monitor", monitor_info.get("Work"))
                result["fullscreen"] = (
                    rect[0] <= mx1 + 2
                    and rect[1] <= my1 + 2
                    and rect[2] >= mx2 - 2
                    and rect[3] >= my2 - 2
                )
        except Exception:
            pass
    except Exception:
        pass
    return result


def sample_gpu_engine_utilization(*, os_name: str) -> dict:
    if os_name != "Windows":
        return {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0}
    command = (
        "$samples=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop).CounterSamples; "
        "$samples | Select-Object Path,CookedValue | ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0}

        rows = json.loads(proc.stdout)
        if isinstance(rows, dict):
            rows = [rows]
        gpu_3d = 0.0
        video_decode = 0.0
        for row in rows:
            path = str(row.get("Path", "")).lower()
            value = float(row.get("CookedValue", 0.0)) / 100.0
            if "engtype_3d" in path:
                gpu_3d += value
            elif "engtype_videodecode" in path or "engtype_video decode" in path:
                video_decode += value
        return {
            "gpu_3d_util": clamp01(gpu_3d),
            "gpu_video_decode_util": clamp01(video_decode),
        }
    except Exception:
        return {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0}


class AutoSignalSampler:
    def __init__(self, *, os_name: str, shutdown_event, interval_s: float = 2.0):
        self.os_name = os_name
        self.shutdown_event = shutdown_event
        self.interval_s = interval_s
        self.lock = threading.Lock()
        self.state = dict(DEFAULT_SIGNAL_STATE)
        self.thread_started = False

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def update(self, samples: dict) -> None:
        with self.lock:
            self.state.update(samples)

    def sample_once(self) -> dict:
        samples = {}
        samples.update(sample_gpu_engine_utilization(os_name=self.os_name))
        samples.update(sample_window_input_context(os_name=self.os_name))
        process = samples.get("foreground_process", "").lower()
        samples["audio_active"] = bool(
            samples.get("gpu_video_decode_util", 0.0) > 0.05
            or any(token in process for token in PLAYER_PROCESS_TOKENS)
        )
        return samples

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            self.update(self.sample_once())
            time.sleep(self.interval_s)
