from stereo_runtime.frame_stats import FrameStats, LatencyStats, format_viewer_title


def test_frame_stats_updates_every_second():
    stats = FrameStats(update_interval=1.0, low_percentile=0.5).start(0.0)

    assert not stats.record_frame(0.5)
    assert stats.record_frame(1.0)
    assert stats.current_fps == 2.0
    assert stats.avg_fps == 2.0
    assert stats.low_fps_avg == 2.0
    assert stats.total_frames == 2


def test_latency_stats_tracks_running_average():
    stats = LatencyStats(history_size=2)

    stats.record(0.1)
    stats.record(0.3)
    stats.record(0.5)

    assert stats.last_display_latency == 0.5
    assert stats.avg_latency == 0.4


def test_format_viewer_title_can_show_detailed_or_compact_stats():
    frame_stats = FrameStats().start(0.0)
    frame_stats.current_fps = 59.6
    frame_stats.avg_fps = 58.0
    frame_stats.low_fps_avg = 45.0
    latency_stats = LatencyStats()
    latency_stats.record(0.012)
    thread_latencies = {
        "capture": 0.001,
        "resize": 0.002,
        "runtime": 0.003,
        "render": 0.004,
    }

    detailed = format_viewer_title(
        frame_stats,
        latency_stats,
        thread_latencies,
        0.004,
        show_fps=True,
    )
    compact = format_viewer_title(
        frame_stats,
        latency_stats,
        thread_latencies,
        0.004,
        show_fps=False,
    )

    assert "59.6FPS" in detailed
    assert "Avg: 58.0" in detailed
    assert "Latency: 12ms" in detailed
    assert compact == "60FPS 12ms"
