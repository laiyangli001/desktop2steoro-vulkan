import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import streaming.direct_sbs as direct_sbs
from streaming.direct_sbs import (
    DirectSbsOutputConsumer,
    FfmpegDirectSbsOutput,
    RuntimeSbsRgbConverter,
    runtime_sbs_to_rgb,
)


def test_runtime_sbs_to_rgb_converts_chw_float_to_hwc_uint8():
    frame = np.array(
        [
            [[0.0, 1.0], [0.5, 0.25]],
            [[1.0, 0.0], [0.5, 0.75]],
            [[0.0, 0.5], [1.0, 0.25]],
        ],
        dtype=np.float32,
    )

    rgb = runtime_sbs_to_rgb(SimpleNamespace(sbs=frame))

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous
    assert rgb[0, 1].tolist() == [255, 0, 128]


def test_runtime_sbs_to_rgb_drops_alpha_from_hwc_uint8():
    frame = np.zeros((2, 3, 4), dtype=np.uint8)
    frame[..., 0] = 7
    frame[..., 3] = 255

    rgb = runtime_sbs_to_rgb(frame)

    assert rgb.shape == (2, 3, 3)
    assert np.all(rgb[..., 0] == 7)


def test_direct_consumer_submits_latest_sbs_frame():
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    submitted = []
    stats = []

    class Output:
        def submit_frame(self, frame):
            submitted.append(frame.copy())
            shutdown.set()

    runtime_q.put((SimpleNamespace(sbs=np.zeros((3, 2, 2))), 1.0))
    runtime_q.put((SimpleNamespace(sbs=np.ones((3, 2, 2))), 2.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
    )

    consumer.run()

    assert len(submitted) == 1
    assert np.all(submitted[0] == 255)
    assert "runtime_output_overwrite" in stats
    assert "network_stream_frames" in stats


def test_direct_consumer_drops_frame_before_conversion_when_not_due():
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    submitted = []

    class Output:
        def should_submit_frame(self, now):
            shutdown.set()
            return False

        def submit_frame(self, frame):
            submitted.append(frame)

    runtime_q.put((SimpleNamespace(sbs=object()), 1.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda *args, **kwargs: None,
    )

    consumer.run()

    assert submitted == []


def test_direct_consumer_logs_fps_when_enabled(capsys):
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    clock_values = iter((0.0, 0.1, 0.2, 0.3, 0.4, 5.0))

    class Output:
        def submit_frame(self, frame):
            shutdown.set()

    runtime_q.put((SimpleNamespace(sbs=np.ones((3, 2, 2))), 1.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda *args, **kwargs: None,
        show_fps_provider=lambda: True,
        clock=lambda: next(clock_values),
    )

    consumer.run()

    assert (
        "[DirectSbsStream] SBS FPS: 0.2 submitted=0.2 "
        "convert_ms=100.0 submit_ms=100.0"
        in capsys.readouterr().out
    )


def test_direct_consumer_hides_fps_when_disabled(capsys):
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(),
        shutdown_event=threading.Event(),
        output=SimpleNamespace(),
        source_stat_inc=lambda *args, **kwargs: None,
        show_fps_provider=lambda: False,
        fps_report_interval=1.0,
        clock=lambda: 0.0,
    )
    consumer._fps_sbs_frames = 30
    consumer._fps_submitted_frames = 29
    consumer._clock = lambda: 1.0

    consumer._report_fps_if_due()

    assert "SBS FPS" not in capsys.readouterr().out


def test_cuda_converter_reuses_pinned_host_buffer():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    converter = RuntimeSbsRgbConverter()
    first_tensor = torch.zeros((3, 2, 4), device="cuda", dtype=torch.uint8)
    second_tensor = torch.full(
        (3, 2, 4), 255, device="cuda", dtype=torch.uint8
    )

    first = converter.convert(first_tensor).copy()
    host_pointer = converter._host_rgb.data_ptr()
    second = converter.convert(second_tensor)

    assert first.shape == (2, 4, 3)
    assert np.all(first == 0)
    assert np.all(second == 255)
    assert converter._host_rgb.data_ptr() == host_pointer
    assert converter._host_rgb.is_pinned()


def test_ffmpeg_output_finds_bundled_encoder_and_config():
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    assert output.ffmpeg_path.name == "ffmpeg.exe"
    assert output.mediamtx_path.name == "mediamtx.exe"
    assert output.mediamtx_config.name == "mediamtx.yml"
    assert output.publish_rtsp_port == 8554
    assert output._server_environment()["MTX_RTMPADDRESS"] == ":1935"


def test_mediamtx_startup_error_includes_server_output(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    class FailedProcess:
        returncode = 1
        stdout = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return (
                "2026/08/19 18:57:14 ERR listen udp :8000: bind: address in use\n",
                None,
            )

    monkeypatch.setattr(
        direct_sbs.subprocess, "Popen", lambda *args, **kwargs: FailedProcess()
    )
    monkeypatch.setattr(direct_sbs.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match=r"listen udp :8000: bind"):
        output.start()


def test_stream_rate_uses_stable_windows_and_fixed_pacing():
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=60,
        crf=20,
        os_name="Windows",
    )

    first_submit_at = None
    for index in range(241):
        timestamp = index / 40.0
        if output.should_submit_frame(timestamp):
            first_submit_at = timestamp
            break

    assert first_submit_at is not None
    assert first_submit_at >= 5.0
    assert output.fps == 30
    assert not output.should_submit_frame(first_submit_at + 0.01)
    assert output.should_submit_frame(first_submit_at + 0.04)


def test_stream_rate_selector_keeps_headroom_across_gpu_speeds():
    select = FfmpegDirectSbsOutput._select_sustainable_stream_fps

    assert select(40.0, 60) == 30
    assert select(28.0, 60) == 25
    assert select(20.0, 60) == 15


def test_unstable_stream_rate_uses_low_recent_percentile():
    sample = FfmpegDirectSbsOutput._fallback_rate_sample

    assert sample([42.0, 35.0, 41.0, 36.0, 40.0, 37.0]) == 36.0


def test_windows_rtmp_command_keeps_legacy_srt_transport_parameters():
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="legacy-compatible",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    command = output._ffmpeg_command(1920, 1080)

    assert command[command.index("-f") + 1] == "rawvideo"
    assert "pipe:0" in command
    assert "gfxcapture" not in " ".join(command)
    assert "libx264" in command
    assert "ultrafast" in command
    assert "zerolatency" in command
    assert "mpegts" in command
    assert (
        "srt://127.0.0.1:8890?"
        "streamid=publish:legacy-compatible&pkt_size=1316"
    ) == command[-1]


def test_nvenc_command_uses_low_latency_hardware_encoder():
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    output.video_encoder = "h264_nvenc"

    command = output._ffmpeg_command(3840, 2160)

    assert "h264_nvenc" in command
    assert "libx264" not in command
    assert command[command.index("-preset") + 1] == "p1"
    assert command[command.index("-cq") + 1] == "20"
    assert command[command.index("-zerolatency") + 1] == "1"
    assert command[command.index("-forced-idr") + 1] == "1"
    assert command[command.index("-strict_gop") + 1] == "1"
    assert "-use_wallclock_as_timestamps" not in command
    assert "-fps_mode" not in command


def test_nvenc_selection_falls_back_when_probe_fails(monkeypatch, capsys):
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    monkeypatch.setattr(output, "_probe_nvenc", lambda width, height: False)

    assert output._select_video_encoder(3840, 2160) == "libx264"
    assert "falling back to libx264" in capsys.readouterr().out


def test_nvenc_probe_uses_actual_sbs_resolution(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(direct_sbs.subprocess, "run", fake_run)

    assert output._probe_nvenc(5120, 1440)
    assert "color=c=black:s=5120x1440:r=1" in commands[0]


def test_rtsp_selection_uses_selected_port_for_internal_publish():
    output = FfmpegDirectSbsOutput(
        base_dir="src",
        protocol="RTSP",
        port=9554,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    assert output.publish_rtsp_port == 9554
    assert output._server_environment()["MTX_RTSPADDRESS"] == ":9554"
