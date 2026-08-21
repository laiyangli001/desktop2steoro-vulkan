from __future__ import annotations

from pathlib import Path

from streaming.vulkan_capabilities import probe_vulkan_video


def test_probe_reports_missing_compiled_encoder(tmp_path: Path) -> None:
    report = probe_vulkan_video(
        tmp_path / "missing-ffmpeg.exe", width=1920, height=1080, os_name="Windows"
    )
    assert not report.available
    assert not report.encoder_compiled
    assert "h264_vulkan" in report.detail


def test_probe_uses_nv12_hwupload_smoke_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, timeout, os_name):
        calls.append(list(command))
        if command[-1] == "-":
            return 0, ""
        return 0, " V....D h264_vulkan Vulkan encoder"

    monkeypatch.setattr("streaming.vulkan_capabilities._run", fake_run)
    report = probe_vulkan_video(tmp_path / "ffmpeg.exe", width=3840, height=2160)
    assert report.available
    assert len(calls) == 2
    assert "format=nv12,hwupload" in calls[1]
    assert "-init_hw_device" in calls[1]
    assert calls[1][calls[1].index("-profile:v") + 1] == "high"
    assert calls[1][calls[1].index("-level:v") + 1] == "5.1"
