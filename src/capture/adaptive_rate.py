from __future__ import annotations

import threading
import time


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
        if sbs_fps < 24.0:
            return min(self.base_fps, 24)
        if sbs_fps < 30.0:
            return min(self.base_fps, 30)
        return self.base_fps

    def _next_target(self, average_sbs_fps: float) -> int:
        if self._target_fps <= 24:
            if average_sbs_fps >= 23.0:
                return min(self.base_fps, 30)
            return min(self.base_fps, 24)
        if self._target_fps <= 30:
            if average_sbs_fps < 24.0:
                return min(self.base_fps, 24)
            if average_sbs_fps >= 29.0:
                return self.base_fps
            return min(self.base_fps, 30)
        return self._bucket_for_sbs(average_sbs_fps)
