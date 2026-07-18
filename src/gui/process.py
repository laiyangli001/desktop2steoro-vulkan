"""GUI Process Mixin — subprocess lifecycle, ESC monitoring, URL actions."""
import os
import re
import sys
import time
import asyncio
import ctypes
import datetime
import json
import logging
import platform
import queue
import subprocess
import traceback
import flet as ft
from utils import OS_NAME, DEFAULT_PORT, shutdown_event, read_yaml
from . import devices as devices_module
from .config import DEFAULTS, default_base_depth_model, save_yaml
from .paths import BASE_DIR, DIAG_LOG, LOG_DIR, LOG_FILE, STOP_REQUEST_FILE
from .capture_sources import get_primary_monitor_index, list_windows
from .localization import UI_MESSAGES
from .log_handler import GuiLogHandler
from utils.logging_setup import _NoisyThirdPartyDebugFilter

# ── module-level console helpers ──

_NOISY_CONSOLE_PREFIXES = (
    "[NativeUtil] sogou_native_util_pc loaded successfully",
    "[warmup] same version",
    "[INFO] [flet] Session was garbage collected:",
)
_DEBUG_CONSOLE_PREFIXES = (
    "[debug]",
    "debug:",
)
_FLET_LOGGER_NAMES = ("flet", "flet_desktop", "flet_controls", "flet_transport")
_FLET_MESSAGE_PREFIXES = ("[flet]", "[flet_desktop]", "[flet_controls]", "[flet_transport]")
_PROGRESS_PERCENT_RE = re.compile(r"(?P<percent>\d{1,3}(?:\.\d+)?)%")
_TQDM_AMOUNT_RE = re.compile(r"\|\s*(?P<completed>[^/\[\]|]+)\s*/\s*(?P<total>[^\[\]|]+)")
_TQDM_TIMING_RE = re.compile(r"\[(?P<elapsed>[^<,\]]+)(?:<(?P<eta>[^,\]]+))?(?:,\s*(?P<speed>[^\]]+))?\]")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_LOG_FILE_LINE_RE = re.compile(r"^\[(?P<asctime>\d\d:\d\d:\d\d)\] \[(?P<level>[A-Z]+)\] \[(?P<name>[^\]]+)\] (?P<message>.*)$")
_LEGACY_LOG_FILE_LINE_RE = re.compile(r"^\[(?P<asctime>\d\d:\d\d:\d\d)\] \[(?P<name>[^\]]+)\] (?P<message>.*)$")
_PROGRESS_PREFIX = "[D2S_PROGRESS] "
_STATUS_PREFIX = "[D2S_STATUS] "
_ASYNCIO_SHUTDOWN_UNRAISABLE_MODULES = (
    "asyncio.base_subprocess",
    "asyncio.proactor_events",
)
_ASYNCIO_SHUTDOWN_UNRAISABLE_MESSAGES = (
    "Event loop is closed",
    "I/O operation on closed pipe",
)
_asyncio_shutdown_noise_filter_installed = False
_console_logging_installed = False
_gui_log_handler = None
logger = logging.getLogger(__name__)
status_logger = logging.getLogger("status")
child_logger = logging.getLogger("child")


def _is_asyncio_shutdown_unraisable(unraisable):
    exc = getattr(unraisable, "exc_value", None)
    if str(exc) not in _ASYNCIO_SHUTDOWN_UNRAISABLE_MESSAGES:
        return False
    obj = getattr(unraisable, "object", None)
    module = getattr(obj, "__module__", "")
    qualname = getattr(obj, "__qualname__", "")
    return module in _ASYNCIO_SHUTDOWN_UNRAISABLE_MODULES and qualname.endswith(".__del__")


def _log_item_from_file_line(line: str):
    text = str(line or "").rstrip("\r\n")
    match = _LOG_FILE_LINE_RE.match(text)
    if match:
        level_name = match.group("level")
        levelno = getattr(logging, level_name, logging.INFO)
        return levelno, match.group("name"), match.group("asctime"), text
    legacy_match = _LEGACY_LOG_FILE_LINE_RE.match(text)
    if legacy_match:
        asctime = legacy_match.group("asctime")
        name = legacy_match.group("name")
        message = legacy_match.group("message")
        return logging.INFO, name, asctime, f"[{asctime}] [INFO] [{name}] {message}"
    return None


def _read_log_file_items():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            return [item for line in file if (item := _log_item_from_file_line(line)) is not None]
    except OSError:
        return []


def _install_asyncio_shutdown_noise_filter():
    """Suppress known Windows asyncio transport __del__ noise during GUI shutdown."""
    global _asyncio_shutdown_noise_filter_installed
    if _asyncio_shutdown_noise_filter_installed or not hasattr(sys, "unraisablehook"):
        return
    previous_hook = sys.unraisablehook

    def _desktop2stereo_unraisable_hook(unraisable):
        if _is_asyncio_shutdown_unraisable(unraisable):
            return
        previous_hook(unraisable)

    sys.unraisablehook = _desktop2stereo_unraisable_hook
    _asyncio_shutdown_noise_filter_installed = True


def _is_key_console_output(data):
    text = str(data or "").strip()
    if not text:
        return True
    lower = text.lower()
    if any(text.startswith(prefix) for prefix in _NOISY_CONSOLE_PREFIXES):
        return False
    if any(lower.startswith(prefix) for prefix in _DEBUG_CONSOLE_PREFIXES):
        return False
    if lower.startswith("[diag]") and not any(token in lower for token in ("error", "failed", "exception", "exited")):
        return False
    return True


class _FletInfoAsDebugFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.INFO and (
            record.name in _FLET_LOGGER_NAMES
            or record.getMessage().startswith(_FLET_MESSAGE_PREFIXES)
        ):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


class _HideStructuredProgressFilter(logging.Filter):
    def filter(self, record):
        return _PROGRESS_PREFIX not in record.getMessage()


class _CpuOperationAsCriticalFilter(logging.Filter):
    def filter(self, record):
        text = f"{record.name} {record.getMessage()}".lower()
        if "cpu" in text:
            record.levelno = logging.CRITICAL
            record.levelname = "CRITICAL"
        return True


def _disable_flet_logging():
    for name in _FLET_LOGGER_NAMES:
        logging.getLogger(name).disabled = True


def _setup_console_logging():
    """Configure console logging, file logging, and GUI log queue."""
    global _console_logging_installed, _gui_log_handler
    _disable_flet_logging()
    if _console_logging_installed:
        _install_asyncio_shutdown_noise_filter()
        return _gui_log_handler

    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        open(LOG_FILE, "w", encoding="utf-8").close()
    except Exception:
        pass

    try:
        for name in os.listdir(LOG_DIR):
            path = os.path.join(LOG_DIR, name)
            if os.path.isfile(path) and os.path.abspath(path) != os.path.abspath(LOG_FILE):
                try:
                    os.remove(path)
                except Exception:
                    pass
    except Exception:
        pass

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console_stream = sys.__stderr__ or sys.stderr or open(os.devnull, "w", encoding="utf-8")
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(_FletInfoAsDebugFilter())
    console_handler.addFilter(_CpuOperationAsCriticalFilter())
    console_handler.addFilter(_NoisyThirdPartyDebugFilter())
    console_handler.addFilter(_HideStructuredProgressFilter())
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
    ))

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(_FletInfoAsDebugFilter())
    file_handler.addFilter(_CpuOperationAsCriticalFilter())
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
    ))

    gui_handler = GuiLogHandler(maxlen=2000)
    gui_handler.setLevel(logging.DEBUG)
    gui_handler.addFilter(_FletInfoAsDebugFilter())
    gui_handler.addFilter(_CpuOperationAsCriticalFilter())
    gui_handler.addFilter(_NoisyThirdPartyDebugFilter())
    gui_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"
    ))

    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.addHandler(gui_handler)
    _gui_log_handler = gui_handler

    class _StreamToLogger:
        def __init__(self, stream_logger, level):
            self.stream_logger = stream_logger
            self.level = level
            self.original = (sys.__stdout__ if level < logging.ERROR else sys.__stderr__) or console_stream
            self._buffer = ""

        def write(self, data):
            if not data:
                return 0
            self._buffer += str(data)
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._log(line.rstrip("\r"))
            return len(data)

        def flush(self):
            if self._buffer.strip():
                self._log(self._buffer.strip())
            self._buffer = ""

        def isatty(self):
            try:
                return self.original.isatty()
            except Exception:
                return False

        def fileno(self):
            return self.original.fileno()

        def _log(self, line):
            if line and _is_key_console_output(line):
                self.stream_logger.log(self.level, line)

    _install_asyncio_shutdown_noise_filter()
    sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR)
    logger.info("Desktop2Stereo log started %s", datetime.datetime.now().isoformat(timespec="seconds"))
    _console_logging_installed = True
    return gui_handler


def _set_console_quick_edit(enabled: bool):
    """Toggle Windows console Quick Edit mode when a real console is attached."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            mode.value |= ENABLE_EXTENDED_FLAGS
            if enabled:
                mode.value |= ENABLE_QUICK_EDIT_MODE
            else:
                mode.value &= ~ENABLE_QUICK_EDIT_MODE
            kernel32.SetConsoleMode(h_stdin, mode)
    except Exception:
        pass


# Disable Quick Edit while the worker is running.
_set_console_quick_edit(False)


class GUIProcessMixin:
    """Mixin providing process lifecycle, ESC monitoring, and status for Desktop2StereoGUI."""

    def set_status(self, msg, key=None):
        self.status_text.value = msg
        if key is not None:
            self._status_key = key
        if msg:
            status_logger.info(msg)
        self._safe_update(self.status_text)

    def _set_running_ui(self, running: bool):
        self.run_btn.disabled = running
        self.stop_btn.disabled = not running
        self._safe_update(self.run_btn, self.stop_btn)

    def _diag(self, msg, error=False):
        os.makedirs(LOG_DIR, exist_ok=True)
        level_name = "ERROR" if error else "INFO"
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        lines = str(msg).splitlines() or [""]
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(f"[{timestamp}] [{level_name}] [diag] {line}\n")
        except Exception:
            pass
        if error:
            try:
                original = getattr(sys.stdout, "original", sys.stdout)
                for line in lines:
                    original.write(f"[{timestamp}] [{level_name}] [diag] {line}\n")
                original.flush()
            except Exception:
                pass

    # ── save & run ──

    def _validate_config_before_run(self):
        try:
            port_val = int(self.stream_port_tf.value) if self.stream_port_tf.value else DEFAULT_PORT
            if not (1 <= port_val <= 65535):
                return False, UI_MESSAGES[self.locale]["Invalid port number (1-65535)"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["Invalid port number (1-65535)"]
        try:
            crf_val = int(self.crf_tf.value) if self.crf_tf.value else DEFAULTS["CRF"]
            if not (0 <= crf_val <= 51):
                return False, UI_MESSAGES[self.locale]["err_crf"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["err_crf"]
        try:
            delay_val = float(self.audio_delay_tf.value) if self.audio_delay_tf.value else DEFAULTS["Audio Delay"]
            if not (-10 <= delay_val <= 10):
                return False, UI_MESSAGES[self.locale]["err_audio_delay"]
        except ValueError:
            return False, UI_MESSAGES[self.locale]["err_audio_delay"]
        sk = self.stream_key_tf.value or "live"
        if not re.match(r'^[A-Za-z0-9_-]+$', sk) or len(sk) > 64:
            return False, UI_MESSAGES[self.locale]["err_stream_key"]
        if self.capture_mode_key == "Window":
            if not self.selected_window_name:
                return False, UI_MESSAGES[self.locale]["Please select a window before running in Window capture mode"]
            windows = list_windows()
            exists = any(
                (w.get("handle") is not None and w["handle"] == self.selected_window_handle)
                or (w.get("handle") is None and w["title"] == self.selected_window_name)
                for w in windows)
            if not exists:
                return False, UI_MESSAGES[self.locale]["The selected window no longer exists. Please refresh and select a valid window."]
        return True, ""

    def save_and_run(self, e):
        if self._starting or (self.process and self.process.returncode is None):
            self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
            self.page.update()
            return
        ok, err = self._validate_config_before_run()
        if not ok:
            self.set_status(err)
            return
        self._starting = True
        self._cancel_starting = False
        self._esc_stopped = False
        self._stopping = False
        _set_console_quick_edit(False)
        self._set_log_panel_visible(self._config.get("Show Log Panel", DEFAULTS["Show Log Panel"]))
        self._set_running_ui(True)
        self._collect_config()
        ok, err = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            self.set_status(UI_MESSAGES[self.locale]["failed_save_yaml"].format(err))
            self._starting = False
            self._set_running_ui(False)
            return
        self.set_status(UI_MESSAGES[self.locale]["Countdown"], key="Countdown")
        self.page.update()
        asyncio.create_task(self._countdown_and_run(0.5))

    async def _countdown_and_run(self, seconds):
        self._diag("_countdown_and_run scheduled")
        try:
            if self.process and self.process.returncode is None:
                self.set_status(UI_MESSAGES[self.locale]["A thread already running!"])
                self._diag("already running, return")
                return
            if seconds > 0:
                await asyncio.sleep(seconds)
            if self._cancel_starting:
                self._cancel_starting = False
                self._diag("cancelled, return")
                return
            status_logger.info(UI_MESSAGES[self.locale].get("Starting Desktop2Stereo...", "Starting Desktop2Stereo...").format(self.run_mode_key))
            shutdown_event.clear()
            try:
                if os.path.exists(STOP_REQUEST_FILE):
                    os.remove(STOP_REQUEST_FILE)
            except Exception:
                pass
            child_args = [
                sys.executable,
                "-u",
                "-X",
                "faulthandler",
                os.path.join(BASE_DIR, "main.py"),
            ]
            child_env = os.environ.copy()
            child_env["DESKTOP2STEREO_LOCALE"] = self.locale
            child_env["PYTHONIOENCODING"] = "utf-8"
            if self.run_mode_key == "OpenXR Link":
                child_env.setdefault("D2S_FPS_BREAKDOWN", "1")
                child_env.setdefault("D2S_OPENXR_DEBUG", "1")
                child_env.setdefault("D2S_OPENXR_ASYNC_EFFECTS", "0")
            if OS_NAME == "Windows":
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    env=child_env,
                )
            else:
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
            self._diag(f"process started, pid={self.process.pid}, log={LOG_FILE}")
            asyncio.create_task(self._pump_child_output(self.process))
            self.set_status(UI_MESSAGES[self.locale]["Running"], key="Running")
            self.page.update()
            asyncio.create_task(self._monitor_process_task())
            self._diag("monitor_task created")
            for _ in range(8):
                await asyncio.sleep(1)
                if self.process and self.process.returncode is not None:
                    self._diag(f"process exited during wait, code={self.process.returncode}")
                    break
            self._config["Recompile TensorRT"] = False
            self._config["Recompile MIGraphX"] = False
            self._config["Recompile CoreML"] = False
            self._config["Recompile OpenVINO"] = False
            save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        except Exception as e:
            self._diag(f"_countdown_and_run failed:\n{traceback.format_exc()}", error=True)
            self.set_status(UI_MESSAGES[self.locale]["err_start_failed"].format(e))
            self.page.update()
        finally:
            self._starting = False

    async def _pump_child_output(self, proc):
        try:
            stream = proc.stdout
            if stream is None:
                return
            pending = ""
            while True:
                raw = await stream.read(4096)
                if not raw:
                    break
                try:
                    pending += raw.decode("utf-8", errors="replace")
                except Exception:
                    pending += repr(raw)
                lines = pending.splitlines(keepends=True)
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines[-1]
                    lines = lines[:-1]
                else:
                    pending = ""
                for line in lines:
                    self._log_child_line(line.rstrip("\r\n"))
            if pending.strip():
                self._log_child_line(pending.strip())
        except Exception as e:
            logger.exception("_pump_child_output exception: %s", e)
            self._diag(f"_pump_child_output exception: {e}\n{traceback.format_exc()}", error=True)

    def _log_child_line(self, line):
        text = str(line or "").strip()
        if not text:
            return
        lower = text.lower()
        if text.startswith(_STATUS_PREFIX):
            status_logger.info(text[len(_STATUS_PREFIX):].strip())
        elif text.startswith("[FPSBreakdown]"):
            child_logger.debug(text)
        elif any(token in lower for token in ("traceback", "exception", "error", "failed", "exited with code")):
            child_logger.error(text)
        elif any(token in lower for token in ("warning", "warn")):
            child_logger.warning(text)
        else:
            child_logger.info(text)

    async def _monitor_process_task(self):
        proc = self.process
        if not proc:
            self._diag("monitor_task: proc is None, return")
            return
        self._diag(f"monitor_task started, pid={proc.pid}")
        try:
            await proc.wait()
            self._diag(f"proc.wait returned, rc={proc.returncode}")
        except Exception as e:
            self._diag(f"proc.wait() exception: {e}", error=True)
        finally:
            self._diag(f"finally: process is proc={self.process is proc}, returncode={proc.returncode}")
            if self.process is proc:
                self.process = None
            self._starting = False
            code = proc.returncode if proc else None
            if code and code != 0:
                self._diag(f"child exited rc={code}; see {LOG_FILE} for details", error=True)
                self.set_status(UI_MESSAGES[self.locale]["exited_with_code"].format(code))
            else:
                self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
            _set_console_quick_edit(True)
            self._set_running_ui(False)
            self._diag("monitor_task done, status updated")

    # ── stop ──

    def stop_process(self, e=None):
        future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    async def _on_page_close(self, e=None):
        self._closed = True
        if hasattr(self, '_esc_task') and self._esc_task and not self._esc_task.done():
            self._esc_task.cancel()
        if hasattr(self, '_log_poll_task') and self._log_poll_task and not self._log_poll_task.done():
            self._log_poll_task.cancel()
        await self._async_stop()

    async def _kill_process_tree(self, proc, pid):
        try:
            proc.kill()
        except Exception:
            pass
        try:
            if OS_NAME == "Windows":
                p = await asyncio.create_subprocess_exec(
                    'taskkill', '/f', '/t', '/pid', str(pid),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await p.wait()
            else:
                import signal
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    pass
        except Exception:
            pass

    async def _async_stop(self):
        if self._stopping:
            if self._closed and self.process and self.process.returncode is None:
                proc = self.process
                self.process = None
                await self._kill_process_tree(proc, proc.pid)
            return
        self._stopping = True
        self._esc_stopped = True
        self._esc_down = None
        self._cancel_starting = True

        if self._proc_lock is not None:
            shutdown_event.set()
            saved_pid = None
            proc = None
            force_kill = False
            async with self._proc_lock:
                proc = self.process
                if proc and proc.returncode is None:
                    saved_pid = proc.pid
                    force_kill = self._closed
                    if not force_kill:
                        try:
                            if OS_NAME == "Windows":
                                os.makedirs(LOG_DIR, exist_ok=True)
                                with open(STOP_REQUEST_FILE, "w", encoding="utf-8") as f:
                                    f.write(str(saved_pid))
                            else:
                                import signal
                                os.killpg(os.getpgid(saved_pid), signal.SIGINT)
                        except Exception:
                            self._diag(f"graceful stop failed:\n{traceback.format_exc()}", error=True)
                            try:
                                proc.terminate()
                            except Exception:
                                self._diag(f"proc.terminate() failed:\n{traceback.format_exc()}", error=True)
                self.process = None

            if saved_pid and proc:
                exited_cleanly = False
                if force_kill:
                    await self._kill_process_tree(proc, saved_pid)
                else:
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1)
                        exited_cleanly = True
                    except asyncio.TimeoutError:
                        exited_cleanly = False
                    except Exception:
                        self._diag(f"proc.wait() exception:\n{traceback.format_exc()}", error=True)
                        exited_cleanly = True
                    if not exited_cleanly:
                        await self._kill_process_tree(proc, saved_pid)
        status_logger.info(UI_MESSAGES[self.locale].get("Runtime stopped", "Stopped"))
        self._starting = False
        self.set_status(UI_MESSAGES[self.locale]["Stopped"], key="Stopped")
        _set_console_quick_edit(True)
        if not self._closed:
            self._set_running_ui(False)

    def _show_log_panel(self):
        panel = getattr(self, "log_panel", None)
        if panel is None:
            return
        panel.visible = True
        if getattr(self, "log_body", None) is not None:
            self.log_body.visible = True
        if getattr(self, "log_toggle_btn", None) is not None:
            self.log_toggle_btn.content.value = "▼"
        if getattr(self, "log_title", None) is not None:
            self.log_title.value = UI_MESSAGES[self.locale].get("Log panel running title", "Running - live log")
            self.log_title.color = None
        self._sync_log_visibility_link()
        self._fit_window_to_content(update=False, resize_window=True)
        self._safe_update(panel, getattr(self, "log_toggle_btn", None), getattr(self, "log_title", None), getattr(self, "log_visibility_link", None))
        try:
            self.page.update()
        except RuntimeError:
            pass

    def _sync_log_visibility_link(self):
        link = getattr(self, "log_visibility_link", None)
        if link is None:
            return
        visible = bool(getattr(getattr(self, "log_panel", None), "visible", False))
        key = "Hide log panel link" if visible else "Show log panel link"
        link.value = UI_MESSAGES[self.locale].get(key, UI_MESSAGES["EN"][key])

    def _set_log_panel_visible(self, visible, save=False, update=True):
        panel = getattr(self, "log_panel", None)
        if panel is None:
            return
        panel.visible = bool(visible)
        self._sync_log_visibility_link()
        self._fit_window_to_content(update=update, resize_window=True)
        if save:
            path = os.path.join(BASE_DIR, "settings.yaml")
            cfg = self._config.copy()
            if os.path.exists(path):
                loaded = read_yaml(path) or {}
                cfg.update(loaded)
            cfg["Show Log Panel"] = panel.visible
            self._config.update(cfg)
            save_yaml(path, cfg)
        if update:
            self._safe_update(panel, getattr(self, "log_visibility_link", None))

    def on_log_visibility_link(self, e=None):
        self._set_log_panel_visible(not getattr(self.log_panel, "visible", False), save=True)
        asyncio.create_task(self._resize_window_after_log_visibility_change())

    async def _resize_window_after_log_visibility_change(self):
        await asyncio.sleep(0)
        self._fit_window_to_content(update=True, resize_window=True)
        await asyncio.sleep(0.5)
        self.page.window.max_width = None
        try:
            self.page.window.update()
        except RuntimeError:
            pass

    def _log_color(self, levelno, text=""):
        lower = str(text or "").lower()
        if levelno >= logging.ERROR:
            return ft.Colors.RED
        if levelno >= logging.WARNING:
            return ft.Colors.ORANGE
        if "100%" in lower or "complete" in lower or "finished" in lower or "cache hit" in lower:
            return ft.Colors.GREEN
        if any(token in lower for token in ("download", "model.safetensors", "model download", "depth model")):
            return ft.Colors.BLUE
        if levelno < logging.INFO:
            return ft.Colors.GREY
        return None

    def _format_gui_log_line(self, item):
        return item[3]

    def _log_emoji(self, logger_name, levelno):
        if logger_name != "status":
            return ""
        if levelno >= logging.ERROR:
            return "❌ "
        if levelno >= logging.WARNING:
            return "⚠️ "
        return "✅ "
    def _selected_log_filter(self):
        return getattr(getattr(self, "log_level_dd", None), "value", "ALL") or "ALL"

    def _log_item_matches_filter(self, item):
        value = self._selected_log_filter()
        levelno, name = item[0], item[1]
        if value == "ALL":
            return True
        if value == "STATUS":
            return name == "status"
        if value == "DEBUG":
            return levelno == logging.DEBUG
        if value == "INFO":
            return levelno == logging.INFO
        if value == "WARNING":
            return levelno == logging.WARNING
        if value == "ERROR":
            return levelno == logging.ERROR
        if value == "CRITICAL":
            return levelno == logging.CRITICAL
        return True

    def _make_log_span(self, item, line=None):
        levelno, name, _, _ = item
        line = self._format_gui_log_line(item) if line is None else line
        color = self._log_color(levelno, line)
        style_kwargs = {
            "size": 12,
            "weight": ft.FontWeight.BOLD if name == "status" else ft.FontWeight.NORMAL,
        }
        if color is not None:
            style_kwargs["color"] = color
        return ft.TextSpan(text=f"{line}\n", style=ft.TextStyle(**style_kwargs))

    def _append_log_span(self, span):
        log_text = getattr(self, "log_text", None)
        if log_text is None:
            return
        if log_text.spans is None:
            log_text.spans = []
        log_text.spans.append(span)
        if len(log_text.spans) > 1000:
            kept_spans = log_text.spans[-500:]
            kept_ids = {id(item) for item in kept_spans}
            log_text.spans = kept_spans
            progress_spans = getattr(self, "_progress_log_spans", {})
            self._progress_log_spans = {
                key: value for key, value in progress_spans.items() if id(value) in kept_ids
            }

    def _progress_event(self, item):
        levelno, name, _, formatted = item
        if levelno >= logging.ERROR or name not in ("child", "stdout"):
            return None
        parsed = _LOG_FILE_LINE_RE.match(formatted)
        message = parsed.group("message") if parsed else formatted
        text = _ANSI_RE.sub("", message).replace("\r", "").strip()
        if _PROGRESS_PREFIX not in text:
            return self._tqdm_progress_event(text)
        text = text[text.index(_PROGRESS_PREFIX):]
        try:
            return json.loads(text[len(_PROGRESS_PREFIX):])
        except Exception:
            return None

    def _tqdm_progress_event(self, text):
        match = _PROGRESS_PERCENT_RE.search(text)
        if not match or "|" not in text or "/" not in text:
            return None
        desc = text[:match.start()].strip(" :-|") or "Progress"
        amount = _TQDM_AMOUNT_RE.search(text)
        timing = _TQDM_TIMING_RE.search(text)
        return {
            "desc": desc,
            "percent": max(0.0, min(100.0, float(match.group("percent")))),
            "downloaded": amount.group("completed").strip() if amount else "",
            "size": amount.group("total").strip() if amount else "",
            "speed": (timing.group("speed") or "").strip() if timing else "",
            "eta": (timing.group("eta") or "00:00").strip() if timing else "",
        }

    def _update_download_progress(self, data):
        panel = getattr(self, "download_progress_panel", None)
        if panel is None:
            return
        percent = data.get("percent")
        desc = str(data.get("desc") or "Download")
        completed = data.get("downloaded") or ""
        total = data.get("size") or ""
        speed = data.get("speed") or ""
        eta = data.get("eta") or ""
        value = 0.0 if percent is None else max(0.0, min(1.0, float(percent) / 100.0))
        done = percent is not None and float(percent) >= 100.0
        color = ft.Colors.GREEN if done else ft.Colors.BLUE
        panel.visible = True
        self.download_progress_title.value = desc
        self.download_progress_title.color = color
        self.download_progress_percent.value = "?" if percent is None else f"{float(percent):.1f}%"
        self.download_progress_percent.color = color
        self.download_progress_bar.value = value
        self.download_progress_bar.color = color
        self.download_progress_detail.value = f"{completed} / {total}  {speed}  ETA {eta}".strip()
        self.download_progress_detail.color = ft.Colors.GREEN if done else ft.Colors.GREY

    def _progress_log_line(self, item):
        levelno, name, _, formatted = item
        if levelno >= logging.WARNING or name not in ("child", "stdout"):
            return None
        marker = "] "
        message = formatted.split(marker, 2)[-1] if marker in formatted else formatted
        text = _ANSI_RE.sub("", message).replace("\r", "").strip()
        match = _PROGRESS_PERCENT_RE.search(text)
        if not match or not any(token in text for token in ("|", "/", "it/s", "ETA", "Downloading", "Exporting", "Building", "Saving", "Runtime preparation")):
            return None
        percent = max(0.0, min(100.0, float(match.group("percent"))))
        desc = text[:match.start()].strip(" :-|")
        if not desc:
            desc = "Progress"
        key = re.sub(r"\s+", " ", desc)
        return key, f"{desc} {percent:5.1f}%"

    def _append_log_item(self, item):
        if not self._log_item_matches_filter(item):
            return
        event = self._progress_event(item)
        if event is not None:
            self._update_download_progress(event)
            return
        progress = self._progress_log_line(item)
        if progress is not None:
            key, line = progress
            progress_spans = getattr(self, "_progress_log_spans", {})
            existing = progress_spans.get(key)
            if existing is not None:
                existing.text = f"{line}\n"
                return
            span = self._make_log_span(item, line)
            progress_spans[key] = span
            self._progress_log_spans = progress_spans
            self._append_log_span(span)
            return
        self._append_log_span(self._make_log_span(item))

    async def _poll_log_queue(self):
        while not self._closed:
            handler = getattr(self, "gui_log_handler", None)
            log_text = getattr(self, "log_text", None)
            if handler is None or log_text is None:
                await asyncio.sleep(0.1)
                continue
            changed = False
            try:
                for _ in range(100):
                    item = handler.queue.get_nowait()
                    try:
                        self._append_log_item(item)
                        if item[0] >= logging.ERROR:
                            self._set_log_problem_state()
                    except Exception:
                        logger.exception("GUI log queue item skipped")
                    changed = True
            except queue.Empty:
                pass
            if changed:
                self._fit_window_to_content(update=False)
                self._safe_update(
                    getattr(self, "log_viewport", None),
                    getattr(self, "log_scroll_row", None),
                    log_text,
                    getattr(self, "download_progress_panel", None),
                    getattr(self, "log_title", None),
                    getattr(self, "report_issue_btn", None),
                )
            await asyncio.sleep(0.1)

    def _set_log_problem_state(self):
        if getattr(self, "log_title", None) is None:
            return
        self.log_title.value = UI_MESSAGES[self.locale].get("Log panel error title", "Issue detected - check logs")
        self.log_title.color = ft.Colors.RED
        if getattr(self, "report_issue_btn", None) is not None:
            self.report_issue_btn.visible = True

    def on_log_toggle(self, e=None):
        self.log_body.visible = not self.log_body.visible
        self.log_toggle_btn.content.value = "▼" if self.log_body.visible else "▶"
        self._fit_window_to_content()
        self._safe_update(self.log_toggle_btn, self.log_body)

    def on_log_level_filter(self, e=None):
        self.log_text.spans = []
        self._progress_log_spans.clear()
        if getattr(self, "download_progress_panel", None) is not None:
            self.download_progress_panel.visible = False
        items = _read_log_file_items()
        handler = getattr(self, "gui_log_handler", None)
        if not items and handler is not None:
            value = getattr(getattr(self, "log_level_dd", None), "value", "ALL")
            items = list(getattr(handler, "status_cache", [])) if value == "STATUS" else list(getattr(handler, "cache", []))
            if value == "ALL" and not any(item[1] == "status" for item in items):
                items.extend(getattr(handler, "status_cache", []))
        for item in items:
            self._append_log_item(item)
        self._safe_update(
            getattr(self, "log_viewport", None),
            getattr(self, "log_scroll_row", None),
            self.log_text,
        )

    def on_log_clear(self, e=None):
        self.log_text.spans = []
        self._progress_log_spans.clear()
        handler = getattr(self, "gui_log_handler", None)
        if handler is not None:
            handler.cache.clear()
            if hasattr(handler, "status_cache"):
                handler.status_cache.clear()
            while True:
                try:
                    handler.queue.get_nowait()
                except queue.Empty:
                    break
        if getattr(self, "log_title", None) is not None:
            self.log_title.value = UI_MESSAGES[self.locale].get("Log panel title", "Run Log")
            self.log_title.color = None
        self._safe_update(
            getattr(self, "log_viewport", None),
            getattr(self, "log_scroll_row", None),
            self.log_text,
            getattr(self, "download_progress_panel", None),
            getattr(self, "log_title", None),
        )

    def on_report_issue(self, e=None):
        handler = getattr(self, "gui_log_handler", None)
        try:
            lines = [
                "=== Desktop2Stereo Bug Report ===",
                f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"OS: {platform.platform()}",
                f"Device: {getattr(getattr(self, 'device_dd', None), 'value', '')}",
                f"Run Mode: {getattr(self, 'run_mode_key', '')}",
                f"Depth Model: {getattr(self, 'current_model_name', '')}",
                "",
                "=== Last log lines ===",
            ]
            if handler is not None:
                for item in list(handler.cache)[-200:]:
                    lines.append(self._format_gui_log_line(item))
            lines.extend(["", "=== Config ===", json.dumps(getattr(self, "_config", {}), indent=2, ensure_ascii=False)])
            text = "\n".join(lines)
            try:
                import pyperclip
                pyperclip.copy(text)
            except ImportError:
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=text, text=True, shell=True)
            self.set_status(UI_MESSAGES[self.locale].get("Bug report copied to clipboard!", "Bug report copied to clipboard!"))
        except Exception as exc:
            logger.exception("Failed to build bug report")
            self.set_status(str(exc))

    def on_open_log_file(self, e=None):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            if not os.path.exists(LOG_FILE):
                open(LOG_FILE, "a", encoding="utf-8").close()
            if OS_NAME == "Windows":
                os.startfile(LOG_FILE)
            elif OS_NAME == "Darwin":
                subprocess.Popen(["open", LOG_FILE])
            else:
                subprocess.Popen(["xdg-open", LOG_FILE])
            self.set_status(UI_MESSAGES[self.locale].get("Opening log file", "Opening log file"))
        except Exception as exc:
            logger.exception("Failed to open log file")
            self.set_status(str(exc))

    # ── reset ──

    def reset_defaults(self, e):
        current_locale = self.locale
        current_device_label = self.device_dd.value
        current_device_idx = self.device_label_to_index.get(current_device_label, DEFAULTS["Computing Device"])
        current_primary = get_primary_monitor_index()
        is_nvidia_cuda = "CUDA" in (current_device_label or "") and not devices_module.IS_ROCM
        dynamic_defaults = DEFAULTS.copy()
        dynamic_defaults["Monitor Index"] = current_primary
        dynamic_defaults["Depth Model"] = default_base_depth_model()
        dynamic_defaults["XR Preview Window"] = False
        if is_nvidia_cuda:
            dynamic_defaults["torch.compile"] = True
            dynamic_defaults["TensorRT"] = True
        self.apply_config(dynamic_defaults, keep_optional=False)
        self.locale = current_locale
        self.lang_dd.value = "English" if current_locale == "EN" else "简体中文"
        self.device_dd.value = current_device_label
        self._config["Language"] = current_locale
        self._config["Computing Device"] = current_device_idx
        self.update_ui_texts()
        self._sync_visibility()
        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.update()

    # ── URL actions ──

    def preview_in_browser(self, e):
        try:
            import webbrowser
            url = self.stream_url_tf.content.controls[0].value
            if not url.startswith(("http://", "https://")):
                self.set_status(UI_MESSAGES[self.locale]["invalid_url_scheme"].format(url))
                return
            webbrowser.open(url)
            self.set_status(f"{UI_MESSAGES[self.locale]['Opening URL in browser']}: {url}")
        except Exception as ex:
            self.set_status(UI_MESSAGES[self.locale]["error_preview"].format(ex))

    def copy_url_to_clipboard(self, e):
        url = self.stream_url_tf.content.controls[0].value
        if url:
            try:
                import pyperclip
                pyperclip.copy(url)
            except ImportError:
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=url, text=True, shell=True)
                elif OS_NAME == "Darwin":
                    subprocess.run("pbcopy", input=url, text=True)
            self.set_status(UI_MESSAGES[self.locale]["url_copied"], key="url_copied")
            asyncio.create_task(self._fade_status(2.0))

    async def _fade_status(self, delay):
        await asyncio.sleep(delay)
        self.set_status("", key="")

    # ── ESC long-press monitoring ──

    VK_ESC = 0x1B

    async def _esc_poll_task(self):
        if OS_NAME != "Windows":
            return
        user32 = ctypes.windll.user32
        try:
            while not self._closed:
                await asyncio.sleep(0.2)
                if self._closed:
                    break
                if user32.GetAsyncKeyState(self.VK_ESC) & 0x8000:
                    if self._esc_down is None:
                        self._esc_down = time.time()
                    elif not self._esc_stopped and (time.time() - self._esc_down >= 3.0):
                        self._esc_stopped = True
                        self._esc_down = None
                        self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                        asyncio.ensure_future(self._async_stop())
                else:
                    if self._esc_down is not None:
                        self._esc_down = None
                        self._esc_stopped = False
        except asyncio.CancelledError:
            pass

    def _on_key(self, e: ft.KeyboardEvent):
        if e.key != "Esc" or self._esc_stopped or OS_NAME == "Windows":
            return
        now = time.time()
        if self._esc_down is None:
            self._esc_down = now
            asyncio.create_task(self._esc_watch_task())
        elif now - self._esc_down >= 3.0:
            self._esc_stopped = True
            self._esc_down = None
            self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
            asyncio.ensure_future(self._async_stop())

    async def _esc_watch_task(self):
        try:
            for _ in range(60):
                await asyncio.sleep(0.05)
                if self._esc_down is None or self._esc_stopped or self._closed:
                    return
                if time.time() - self._esc_down >= 3.0:
                    self._esc_stopped = True
                    self._esc_down = None
                    self.set_status(UI_MESSAGES[self.locale]["esc_stop"])
                    asyncio.ensure_future(self._async_stop())
                    return
        except asyncio.CancelledError:
            pass
