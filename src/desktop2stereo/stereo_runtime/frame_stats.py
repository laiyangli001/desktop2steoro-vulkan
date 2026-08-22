from __future__ import annotations

from collections import deque


class FrameStats:
    def __init__(self, *, history_size=300, update_interval=5.0, low_percentile=0.01):
        self.frame_count = 0
        self.total_frames = 0
        self.start_time = None
        self.last_time = None
        self.last_update_time = None
        self.current_fps = 0.0
        self.avg_fps = 0.0
        self.low_fps_avg = float("inf")
        self.history_size = history_size
        self.update_interval = update_interval
        self.low_percentile = low_percentile
        self.fps_values = deque(maxlen=history_size)

    def start(self, now):
        self.start_time = now
        self.last_time = now
        self.last_update_time = now
        return self

    def record_frame(self, now):
        if self.start_time is None:
            self.start(now)
        self.frame_count += 1
        self.total_frames += 1
        elapsed = now - self.last_time
        if elapsed < 1.0:
            return False

        self.current_fps = self.frame_count / elapsed
        self.frame_count = 0
        self.last_time = now
        self.fps_values.append(self.current_fps)

        if now - self.last_update_time >= self.update_interval:
            self._refresh_summary()
            self.last_update_time = now
        return True

    def _refresh_summary(self):
        if not self.fps_values:
            return
        self.avg_fps = sum(self.fps_values) / len(self.fps_values)
        sorted_fps = sorted(self.fps_values)
        low_count = int(len(sorted_fps) * self.low_percentile)
        if low_count == 0:
            low_count = 1
        low_values = sorted_fps[:low_count]
        self.low_fps_avg = sum(low_values) / len(low_values)

    def overall_avg_fps(self, now):
        if self.start_time is None:
            return 0.0
        total_time = now - self.start_time
        return self.total_frames / total_time if total_time > 0 else 0.0


class LatencyStats:
    def __init__(self, *, history_size=300):
        self.history_size = history_size
        self.history = deque()
        self.total = 0.0
        self.avg_latency = 0.0
        self.last_display_latency = 0.0

    def record(self, latency):
        self.history.append(latency)
        self.total += latency
        if len(self.history) > self.history_size:
            self.total -= self.history.popleft()
        self.last_display_latency = latency
        if self.history:
            self.avg_latency = self.total / len(self.history)


def format_viewer_title(stats, latency_stats, thread_latencies, render_latency, *, show_fps):
    if not show_fps:
        return f"{stats.current_fps:.0f}FPS {latency_stats.last_display_latency * 1000:.0f}ms"
    return (
        f"{stats.current_fps:.1f}FPS | "
        f"Avg: {stats.avg_fps:.1f} | "
        f"1% Low Avg: {stats.low_fps_avg:.1f} | "
        f"Latency: {latency_stats.last_display_latency * 1000:.0f}ms | "
        f"Avg Latency: {latency_stats.avg_latency * 1000:.0f}ms "
        f"(Capture:{thread_latencies['capture'] * 1000:.0f}ms "
        f"Resize:{thread_latencies['resize'] * 1000:.0f}ms "
        f"Runtime:{thread_latencies['runtime'] * 1000:.0f}ms "
        f"Render:{render_latency * 1000:.0f}ms)"
    )
