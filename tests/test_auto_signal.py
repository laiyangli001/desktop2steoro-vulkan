from __future__ import annotations

import threading

from stereo_runtime import auto_signal


def test_non_windows_samples_return_defaults():
    assert auto_signal.query_process_name(123, os_name="Linux") == ""
    assert auto_signal.sample_window_input_context(os_name="Linux") == {
        "input_activity": 0.0,
        "idle_seconds": 0.0,
        "maximized": False,
        "foreground_process": "",
        "fullscreen": False,
    }
    assert auto_signal.sample_gpu_engine_utilization(os_name="Linux") == {
        "gpu_3d_util": 0.0,
        "gpu_video_decode_util": 0.0,
    }


def test_auto_signal_sampler_sets_audio_active_from_video_decode(monkeypatch):
    monkeypatch.setattr(
        auto_signal,
        "sample_gpu_engine_utilization",
        lambda *, os_name: {"gpu_3d_util": 0.1, "gpu_video_decode_util": 0.2},
    )
    monkeypatch.setattr(
        auto_signal,
        "sample_window_input_context",
        lambda *, os_name: {"foreground_process": ""},
    )

    sampler = auto_signal.AutoSignalSampler(os_name="Windows", shutdown_event=threading.Event())

    assert sampler.sample_once()["audio_active"] is True


def test_auto_signal_sampler_sets_audio_active_from_process(monkeypatch):
    monkeypatch.setattr(
        auto_signal,
        "sample_gpu_engine_utilization",
        lambda *, os_name: {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0},
    )
    monkeypatch.setattr(
        auto_signal,
        "sample_window_input_context",
        lambda *, os_name: {"foreground_process": "chrome.exe"},
    )

    sampler = auto_signal.AutoSignalSampler(os_name="Windows", shutdown_event=threading.Event())

    assert sampler.sample_once()["audio_active"] is True


def test_auto_signal_snapshot_is_copy():
    sampler = auto_signal.AutoSignalSampler(os_name="Linux", shutdown_event=threading.Event())
    sampler.update({"gpu_3d_util": 0.5})

    snapshot = sampler.snapshot()
    snapshot["gpu_3d_util"] = 0.0

    assert sampler.snapshot()["gpu_3d_util"] == 0.5
