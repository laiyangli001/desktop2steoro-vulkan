from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


DEFAULT_SOURCE_STATS = {
    "capture_frames": 0,
    "capture_errors": 0,
    "capture_dropped_paused": 0,
    "raw_put": 0,
    "raw_get": 0,
    "raw_queue_empty": 0,
    "runtime_frames": 0,
    "runtime_none": 0,
    "runtime_errors": 0,
    "runtime_dropped_paused": 0,
    "last_capture_ts": 0.0,
    "last_raw_get_ts": 0.0,
    "last_runtime_ts": 0.0,
    "last_process_latency": 0.0,
    "last_runtime_latency": 0.0,
    "last_error": "",
}


def safe_qsize(q) -> int:
    try:
        return q.qsize()
    except Exception:
        return -1


def format_age(seconds: float) -> str:
    if seconds < 0:
        return "n/a"
    return f"{seconds:.2f}s"


class SourceHealth:
    def __init__(
        self,
        *,
        enabled: bool,
        run_mode: str,
        raw_q,
        runtime_q,
        source_active: Callable[[], bool],
        render_active: Callable[[], bool],
        idle_active: Callable[[], bool],
    ):
        self.enabled = enabled
        self.run_mode = run_mode
        self.raw_q = raw_q
        self.runtime_q = runtime_q
        self.source_active = source_active
        self.render_active = render_active
        self.idle_active = idle_active
        self.lock = threading.Lock()
        self.stats = dict(DEFAULT_SOURCE_STATS)
        self.last_log = 0.0
        self.log_count = 0
        self.log_limit = 5

    def inc(self, name: str, amount: int | float = 1, **values) -> None:
        with self.lock:
            self.stats[name] = self.stats.get(name, 0) + amount
            self.stats.update(values)

    def set(self, **values) -> None:
        with self.lock:
            self.stats.update(values)

    def log(self, now: float | None = None, force: bool = False) -> None:
        if self.run_mode != "OpenXR":
            return
        if not self.enabled:
            return
        if self.log_count >= self.log_limit:
            return
        now = time.perf_counter() if now is None else now
        if not force and (now - self.last_log) < 5.0:
            return
        self.last_log = now
        self.log_count += 1
        with self.lock:
            stats = dict(self.stats)

        last_capture = stats.get("last_capture_ts", 0.0)
        last_runtime = stats.get("last_runtime_ts", 0.0)
        raw_age = now - last_capture if last_capture > 0.0 else -1.0
        runtime_age = now - last_runtime if last_runtime > 0.0 else -1.0
        last_error = stats.get("last_error") or "none"
        logger.debug(
            "[Main] Source health: "
            f"cap={stats.get('capture_frames', 0)} raw_put={stats.get('raw_put', 0)} "
            f"raw_get={stats.get('raw_get', 0)} runtime={stats.get('runtime_frames', 0)} "
            f"empty={stats.get('raw_queue_empty', 0)} none={stats.get('runtime_none', 0)} "
            f"cap_err={stats.get('capture_errors', 0)} runtime_err={stats.get('runtime_errors', 0)} "
            f"raw_age={format_age(raw_age)} runtime_age={format_age(runtime_age)} "
            f"raw_q={safe_qsize(self.raw_q)} runtime_q={safe_qsize(self.runtime_q)} "
            f"resize_ms={stats.get('last_process_latency', 0.0) * 1000.0:.1f} "
            f"runtime_ms={stats.get('last_runtime_latency', 0.0) * 1000.0:.1f} "
            f"source={self.source_active()} render={self.render_active()} "
            f"idle={self.idle_active()} err={last_error}"
        )
