from __future__ import annotations

import math
import threading
import time


def adaptive_capture_enabled_for_mode(run_mode: str, target_fps: int) -> bool:
    mode = str(run_mode or "").strip().lower()
    return int(target_fps) <= 0 and mode in {
        "local viewer",
        "viewer",
        "openxr",
        "openxr link",
    }


class AdaptiveCaptureRate:
    """Select a bounded capture rate from low-rate SBS throughput samples."""

    def __init__(
        self,
        base_fps: int,
        *,
        enabled: bool,
        evaluation_interval_s: float = 60.0,
    ) -> None:
        self.base_fps = max(1, int(base_fps))
        self.enabled = bool(enabled and self.base_fps > 24)
        self.evaluation_interval_s = max(1.0, float(evaluation_interval_s))
        self._target_fps = self.base_fps
        self._window_started: float | None = None
        self._window_samples: list[float] = []
        self._lock = threading.Lock()

    def current_fps(self) -> int:
        with self._lock:
            return int(self._target_fps)

    def observe_sbs_fps(self, sbs_fps: float, *, now: float | None = None) -> int:
        timestamp = time.monotonic() if now is None else float(now)
        try:
            measured = max(0.0, float(sbs_fps))
        except (TypeError, ValueError):
            return self.current_fps()
        with self._lock:
            if not self.enabled or measured <= 0.0:
                return int(self._target_fps)
            if self._window_started is None:
                self._window_started = timestamp
                self._window_samples = [measured]
                return int(self._target_fps)
            self._window_samples.append(measured)
            if timestamp - self._window_started < self.evaluation_interval_s:
                return int(self._target_fps)
            average = sum(self._window_samples) / max(1, len(self._window_samples))
            self._target_fps = self._next_target(average)
            self._window_started = timestamp
            self._window_samples.clear()
            return int(self._target_fps)

    def _bucket_for_sbs(self, sbs_fps: float) -> int:
        measured = max(0.0, float(sbs_fps))
        # Keep only a small capture headroom above the measured SBS rate.
        # Capture backends accept arbitrary integer pacing targets, so fixed
        # display-refresh buckets would waste work (for example 60 -> 120).
        return min(self.base_fps, max(1, int(math.ceil(measured + 5.0))))

    def _next_target(self, average_sbs_fps: float) -> int:
        # The averaging window already provides hysteresis. Keep only 5 FPS
        # headroom instead of jumping to the monitor refresh rate.
        return self._bucket_for_sbs(average_sbs_fps)
