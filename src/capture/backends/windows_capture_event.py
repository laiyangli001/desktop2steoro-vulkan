from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes

from capture.types import FrameCopyMode, capture_frame_from_raw


CAPTURE_CURSOR_DELAY_S = 0.2
CAPTURE_GAP_LOG_LIMIT = 3
_PENDING_CAPTURE_GAP_LOGS = []
_PENDING_CAPTURE_GAP_LOCK = threading.Lock()
_CAPTURE_GAP_DEFER_RELEASED = False


def _env_bool(name):
    value = os.environ.get(name)
    if value is None:
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _emit_capture_gap_log(line: str) -> None:
    print(line, flush=True)


def flush_pending_capture_gap_logs() -> None:
    global _CAPTURE_GAP_DEFER_RELEASED
    with _PENDING_CAPTURE_GAP_LOCK:
        pending = list(_PENDING_CAPTURE_GAP_LOGS)
        _PENDING_CAPTURE_GAP_LOGS.clear()
        _CAPTURE_GAP_DEFER_RELEASED = True
    for line in pending:
        _emit_capture_gap_log(line)


def _defer_capture_gap_log(line: str) -> bool:
    if not _env_bool("D2S_DEFER_CAPTURE_GAP_UNTIL_OPENXR_PROJECTION"):
        return False
    with _PENDING_CAPTURE_GAP_LOCK:
        if _CAPTURE_GAP_DEFER_RELEASED:
            return False
        _PENDING_CAPTURE_GAP_LOGS.append(line)
    return True


def _fps_to_minimum_update_interval_ms(fps):
    try:
        fps_value = int(fps)
    except (TypeError, ValueError):
        return None
    if fps_value <= 0:
        return None
    return max(1, int(round(1000.0 / fps_value)))


def _windows_capture_kwargs(config, capture_tool):
    if config.capture_mode == "Window":
        kwargs = {"window_name": config.window_title}
    else:
        kwargs = {"monitor_index": config.monitor_index}
    if capture_tool == "WindowsCaptureCUDA":
        optional = {
            "minimum_update_interval": _fps_to_minimum_update_interval_ms(getattr(config, "fps", None)),
            "reuse_output_buffer": _env_bool("D2S_WGC_REUSE_OUTPUT_BUFFER"),
            "output_buffer_count": _env_int("D2S_WGC_OUTPUT_BUFFER_COUNT"),
        }
        kwargs.update({key: value for key, value in optional.items() if value is not None})
    return kwargs


def _load_windows_capture(capture_tool):
    if capture_tool == "WindowsCaptureROCm":
        from wc_rocm import WindowsCapture, Frame, InternalCaptureControl
    elif capture_tool == "WindowsCaptureCUDA":
        from wc_cuda import WindowsCapture, Frame, InternalCaptureControl
    else:
        from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    return WindowsCapture, Frame, InternalCaptureControl



def _event_capture_device(capture_tool):
    if capture_tool == "WindowsCaptureCUDA":
        return "cuda"
    if capture_tool == "WindowsCaptureROCm":
        return "rocm"
    return "cpu"


def _copy_frame_buffer(frame_buffer, capture_tool):
    device = _event_capture_device(capture_tool)
    if device in ("cuda", "rocm") and not _env_bool("D2S_WGC_COPY_FRAME_BUFFER"):
        return frame_buffer, FrameCopyMode.GPU_TENSOR, device
    prefer_clone = device in ("cuda", "rocm")
    if prefer_clone and hasattr(frame_buffer, "clone"):
        return frame_buffer.clone(), FrameCopyMode.CLONE, device
    if hasattr(frame_buffer, "copy"):
        return frame_buffer.copy(), FrameCopyMode.COPY, device
    return frame_buffer.clone(), FrameCopyMode.CLONE, device


def _setup_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()


class WindowsCaptureEventRunner:
    def __init__(self, config):
        self.config = config
        self.capture_tool = config.capture_tool or "WindowsCapture"
        self._capture_started_event = threading.Event()
        self._keyboard_thread = None
        self._session = None
        self._control = None
        self._fps_last_log = 0.0
        self._fps_frames = 0
        self._fps_copy_seconds = 0.0
        self._fps_enqueue_seconds = 0.0
        self._fps_handler_seconds = 0.0
        self._last_frame_ts = 0.0
        self._capture_gap_logs = 0

    @property
    def session(self):
        return self._session

    @property
    def control(self):
        return self._control

    def stop(self):
        if self._control is not None:
            self._control.stop()
        elif self._session is not None and hasattr(self._session, "stop"):
            self._session.stop()

    def _log_capture_fps(self, now: float) -> None:
        if self.capture_tool != "WindowsCaptureCUDA":
            return
        if not _env_bool("D2S_WGC_CAPTURE_FPS_LOG"):
            return
        if self._fps_last_log <= 0.0:
            self._fps_last_log = now
            self._fps_frames = 0
            return
        self._fps_frames += 1
        elapsed = now - self._fps_last_log
        if elapsed < 1.0:
            return
        fps = self._fps_frames / elapsed
        copy_ms = self._fps_copy_seconds * 1000.0 / max(self._fps_frames, 1)
        enqueue_ms = self._fps_enqueue_seconds * 1000.0 / max(self._fps_frames, 1)
        handler_ms = self._fps_handler_seconds * 1000.0 / max(self._fps_frames, 1)
        print(
            f"[WindowsCaptureCUDA] capture_fps={fps:.1f} frames={self._fps_frames} "
            f"monitor={self.config.monitor_index} mode={self.config.capture_mode} "
            f"copy_ms={copy_ms:.2f} enqueue_ms={enqueue_ms:.2f} handler_ms={handler_ms:.2f}",
            flush=True,
        )
        self._fps_last_log = now
        self._fps_frames = 0
        self._fps_copy_seconds = 0.0
        self._fps_enqueue_seconds = 0.0
        self._fps_handler_seconds = 0.0

    def _record_capture_timing(self, *, copy_seconds: float, enqueue_seconds: float, handler_seconds: float) -> None:
        if self.capture_tool != "WindowsCaptureCUDA":
            return
        if not _env_bool("D2S_WGC_CAPTURE_FPS_LOG"):
            return
        self._fps_copy_seconds += copy_seconds
        self._fps_enqueue_seconds += enqueue_seconds
        self._fps_handler_seconds += handler_seconds

    def _log_capture_gap(self, now: float, capture_kwargs: dict) -> None:
        if self.capture_tool != "WindowsCaptureCUDA":
            self._last_frame_ts = now
            return
        gap = now - self._last_frame_ts if self._last_frame_ts > 0.0 else 0.0
        self._last_frame_ts = now
        if gap < 0.5:
            return
        if self._capture_gap_logs >= CAPTURE_GAP_LOG_LIMIT:
            return
        self._capture_gap_logs += 1
        line = (
            f"[CaptureGap] tool={self.capture_tool} mode={self.config.capture_mode} "
            f"monitor={self.config.monitor_index} gap={gap:.2f}s kwargs={capture_kwargs}"
        )
        if not _defer_capture_gap_log(line):
            _emit_capture_gap_log(line)

    def _start_keyboard_worker(self, shutdown_event):
        user32 = ctypes.windll.user32
        user32.ShowCursor.argtypes = [wintypes.BOOL]
        user32.ShowCursor.restype = ctypes.c_int
        user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_ulonglong]
        user32.keybd_event.restype = None

        vk_menu = 0x12
        vk_tab = 0x09
        keyeventf_keyup = 0x0002

        def simulate_alt_tab():
            user32.keybd_event(vk_menu, 0, 0, 0)
            user32.keybd_event(vk_tab, 0, 0, 0)
            time.sleep(0.01)
            user32.keybd_event(vk_tab, 0, keyeventf_keyup, 0)
            user32.keybd_event(vk_menu, 0, keyeventf_keyup, 0)
            return True

        def keyboard_worker():
            while not shutdown_event.is_set():
                triggered = self._capture_started_event.wait(timeout=0.1)
                if shutdown_event.is_set():
                    break
                if not triggered:
                    continue
                try:
                    simulate_alt_tab()
                    time.sleep(0.2)
                    simulate_alt_tab()
                    if CAPTURE_CURSOR_DELAY_S:
                        time.sleep(CAPTURE_CURSOR_DELAY_S)
                except Exception as exc:
                    print(f"[keyboard] Exception during action: {exc}")
                finally:
                    break

        self._keyboard_thread = threading.Thread(target=keyboard_worker, name="CursorWorker", daemon=True)
        self._keyboard_thread.start()

    def run(
        self,
        *,
        shutdown_event,
        on_frame,
        on_error=None,
        on_closed=None,
        is_paused=None,
        is_hard_idle=None,
        on_paused=None,
        on_session_update=None,
        on_tick=None,
    ):
        _setup_dpi_awareness()
        WindowsCapture, Frame, InternalCaptureControl = _load_windows_capture(self.capture_tool)
        self._start_keyboard_worker(shutdown_event)

        while not shutdown_event.is_set():
            if on_tick is not None:
                on_tick()
            if is_hard_idle is not None and is_hard_idle():
                if on_paused is not None:
                    on_paused("hard_idle")
                time.sleep(0.1)
                continue

            capture_kwargs = _windows_capture_kwargs(self.config, self.capture_tool)
            if self.config.capture_mode != "Window":
                if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                    print(
                        f"[capture_loop] WindowsCapture monitor_index={self.config.monitor_index} "
                        f"tool={self.capture_tool} kwargs={capture_kwargs}",
                        flush=True,
                    )
            cap = WindowsCapture(**capture_kwargs)
            self._session = cap
            self._control = None
            if on_session_update is not None:
                on_session_update(self._session, self._control)

            @cap.event
            def on_frame_arrived(frame: Frame, internal_capture_control: InternalCaptureControl):
                self._control = internal_capture_control
                if on_session_update is not None:
                    on_session_update(self._session, self._control)
                capture_start_time = time.perf_counter()
                if shutdown_event.is_set():
                    return
                if (is_hard_idle is not None and is_hard_idle()) or (is_paused is not None and is_paused()):
                    if on_paused is not None:
                        on_paused("paused")
                    return
                self._log_capture_gap(capture_start_time, capture_kwargs)
                self._log_capture_fps(capture_start_time)
                copy_start_time = time.perf_counter()
                raw, copy_mode, frame_raw_device = _copy_frame_buffer(frame.frame_buffer, self.capture_tool)
                enqueue_start_time = time.perf_counter()
                on_frame(
                    capture_frame_from_raw(
                        raw,
                        self.config.output_resolution,
                        capture_start_time,
                        config=self.config,
                        copy_mode=copy_mode,
                        original_format=type(frame.frame_buffer).__name__,
                        frame_raw_device=frame_raw_device,
                        metadata={
                            "backend": "windows_capture_event",
                            "zero_copy": copy_mode is FrameCopyMode.GPU_TENSOR,
                        },
                    )
                )
                handler_end_time = time.perf_counter()
                self._record_capture_timing(
                    copy_seconds=enqueue_start_time - copy_start_time,
                    enqueue_seconds=handler_end_time - enqueue_start_time,
                    handler_seconds=handler_end_time - capture_start_time,
                )

            @cap.event
            def closed():
                if on_closed is not None:
                    on_closed()

            try:
                cap.start()
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                else:
                    raise
                time.sleep(0.5)
            finally:
                self._control = None
                self._session = None
                if on_session_update is not None:
                    on_session_update(None, None)

            if shutdown_event.is_set():
                break
            time.sleep(0.1)
