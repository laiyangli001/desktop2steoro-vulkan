from __future__ import annotations

import importlib
import math
import os
import re
from pathlib import Path
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

from streaming.encoder_profile import EncoderProfile
from streaming.mjpeg_streamer import MJPEGStreamer
from streaming.runtime_manager import ensure_runtime
from streaming.nvidia_encoder import PyNvSrtVideoOutput, PyNvVideoCodecEncoder
from streaming.stream_calibration import StreamCalibrationController
from streaming.wasapi_audio import SoundcardLoopbackSender


_PYNVVIDEO_CODEC = None
_PYNVVIDEO_CODEC_ERROR: str | None = None
_PYNVVIDEO_DLL_HANDLES: list[Any] = []


def _load_pynvvideo_codec() -> Any | None:
    """Load PyNvVideoCodec with bundled CUDA runtime DLLs when available."""
    global _PYNVVIDEO_CODEC, _PYNVVIDEO_CODEC_ERROR
    if _PYNVVIDEO_CODEC is not None:
        return _PYNVVIDEO_CODEC
    if _PYNVVIDEO_CODEC_ERROR is not None:
        return None
    try:
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            candidates = []
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                candidates.append(Path(cuda_path) / "bin")
            candidates.extend(
                [
                    Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
                    Path(__file__).resolve().parents[2] / "python3" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
                ]
            )
            for path in candidates:
                if path.is_dir():
                    _PYNVVIDEO_DLL_HANDLES.append(os.add_dll_directory(str(path)))
        _PYNVVIDEO_CODEC = importlib.import_module("PyNvVideoCodec")
        return _PYNVVIDEO_CODEC
    except Exception as exc:
        _PYNVVIDEO_CODEC_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def runtime_sbs_to_rgb(frame_or_result: Any) -> np.ndarray:
    """Convert a packed SBS runtime tensor/array to contiguous RGB8 HWC."""
    frame = getattr(frame_or_result, "sbs", frame_or_result)
    if frame is None:
        raise ValueError("runtime result does not contain an SBS frame")
    image = frame.detach() if hasattr(frame, "detach") else frame
    if bool(getattr(image, "is_cuda", False)):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"unsupported SBS frame shape: {image.shape!r}")
    if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"unsupported SBS channel count: {image.shape[-1]}")
    if image.dtype != np.uint8:
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


class RuntimeSbsRgbConverter:
    """Convert runtime SBS frames with a reusable pinned CUDA download buffer."""

    def __init__(self, *, copy_output: bool = False) -> None:
        self.copy_output = bool(copy_output)
        self._host_rgb = None

    def convert(self, frame_or_result: Any) -> np.ndarray:
        frame = getattr(frame_or_result, "sbs", frame_or_result)
        if frame is None:
            raise ValueError("runtime result does not contain an SBS frame")
        image = frame.detach() if hasattr(frame, "detach") else frame
        if not bool(getattr(image, "is_cuda", False)):
            return runtime_sbs_to_rgb(image)
        return self._cuda_to_rgb(image)

    def _cuda_to_rgb(self, image) -> np.ndarray:
        import torch

        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise ValueError(
                    f"expected one SBS frame, got shape {tuple(image.shape)!r}"
                )
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"unsupported SBS frame shape: {tuple(image.shape)!r}")

        if int(image.shape[0]) in (1, 3, 4):
            channels = int(image.shape[0])
            if channels == 1:
                image = image.expand(3, -1, -1)
            elif channels == 4:
                image = image[:3]
            image = image.permute(1, 2, 0)
        elif int(image.shape[-1]) in (1, 3, 4):
            channels = int(image.shape[-1])
            if channels == 1:
                image = image.expand(-1, -1, 3)
            elif channels == 4:
                image = image[..., :3]
        else:
            raise ValueError(
                f"unsupported SBS channel count for shape {tuple(image.shape)!r}"
            )

        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        image = image.contiguous()
        if (
            self._host_rgb is None
            or tuple(self._host_rgb.shape) != tuple(image.shape)
        ):
            self._host_rgb = torch.empty(
                tuple(image.shape),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
        self._host_rgb.copy_(image, non_blocking=True)
        torch.cuda.current_stream(device=image.device).synchronize()
        result = self._host_rgb.numpy()
        return result.copy() if self.copy_output else result


class DirectSbsOutputConsumer:
    """Consume only the newest runtime SBS frame and submit it to a stream sink."""

    def __init__(
        self,
        *,
        runtime_q,
        shutdown_event,
        output,
        source_stat_inc: Callable[..., None],
        show_fps_provider: Callable[[], bool] | None = None,
        on_sbs_fps: Callable[..., Any] | None = None,
        fps_report_interval: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.runtime_q = runtime_q
        self.shutdown_event = shutdown_event
        self.output = output
        self.source_stat_inc = source_stat_inc
        self.show_fps_provider = show_fps_provider
        self.on_sbs_fps = on_sbs_fps
        self.fps_report_interval = max(0.1, float(fps_report_interval))
        self._clock = clock
        self._fps_started = self._clock()
        self._fps_sbs_frames = 0
        self._fps_submitted_frames = 0
        self._fps_convert_seconds = 0.0
        self._fps_submit_seconds = 0.0
        self._frame_converter = RuntimeSbsRgbConverter(
            copy_output=not bool(getattr(output, "synchronous_submit", False))
        )

    def _take_latest(self):
        try:
            item = self.runtime_q.get(timeout=0.05)
        except queue.Empty:
            return None
        while True:
            try:
                item = self.runtime_q.get_nowait()
                self.source_stat_inc("runtime_output_overwrite")
            except queue.Empty:
                return item

    def _report_fps_if_due(self) -> None:
        now = self._clock()
        elapsed = now - self._fps_started
        if elapsed < self.fps_report_interval:
            return
        sbs_fps = self._fps_sbs_frames / elapsed
        submitted_fps = self._fps_submitted_frames / elapsed
        convert_ms = (
            self._fps_convert_seconds * 1000.0 / self._fps_sbs_frames
            if self._fps_sbs_frames
            else 0.0
        )
        submit_ms = (
            self._fps_submit_seconds * 1000.0 / self._fps_submitted_frames
            if self._fps_submitted_frames
            else 0.0
        )
        if self.on_sbs_fps is not None:
            self.on_sbs_fps(sbs_fps, frame_count=self._fps_sbs_frames)
        observe_calibration = getattr(self.output, "observe_calibration_window", None)
        if callable(observe_calibration):
            observe_calibration(
                sbs_fps=sbs_fps,
                submitted_fps=submitted_fps,
                convert_ms=convert_ms,
                submit_ms=submit_ms,
            )
        show_fps = (
            bool(self.show_fps_provider())
            if self.show_fps_provider is not None
            else False
        )
        if show_fps:
            print(
                f"[DirectSbsStream] SBS FPS: {sbs_fps:.1f} "
                f"submitted={submitted_fps:.1f} "
                f"convert_ms={convert_ms:.1f} submit_ms={submit_ms:.1f}",
                flush=True,
            )
        self._fps_sbs_frames = 0
        self._fps_submitted_frames = 0
        self._fps_convert_seconds = 0.0
        self._fps_submit_seconds = 0.0
        self._fps_started = now

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            item = self._take_latest()
            if item is None:
                continue
            try:
                runtime_result, _capture_timestamp = item
                self._fps_sbs_frames += 1
                prepare_calibration = getattr(
                    self.output, "prepare_calibration_source", None
                )
                if callable(prepare_calibration) and prepare_calibration(runtime_result):
                    self._fps_submitted_frames += 1
                    self.source_stat_inc("runtime_output_frames")
                    self.source_stat_inc("network_stream_frames")
                    self._report_fps_if_due()
                    continue
                should_submit = getattr(self.output, "should_submit_frame", None)
                if callable(should_submit) and not should_submit(self._clock()):
                    self._report_fps_if_due()
                    continue
                submit_cuda_frame = getattr(self.output, "submit_cuda_frame", None)
                cuda_frame = getattr(runtime_result, "sbs", runtime_result)
                if callable(submit_cuda_frame) and bool(
                    getattr(cuda_frame, "is_cuda", False)
                ):
                    convert_started = self._clock()
                    submit_cuda_frame(cuda_frame)
                    self._fps_submit_seconds += self._clock() - convert_started
                else:
                    convert_started = self._clock()
                    frame = self._frame_converter.convert(runtime_result)
                    self._fps_convert_seconds += self._clock() - convert_started
                    submit_started = self._clock()
                    self.output.submit_frame(frame)
                    self._fps_submit_seconds += self._clock() - submit_started
                self._fps_submitted_frames += 1
                self.source_stat_inc("runtime_output_frames")
                self.source_stat_inc("network_stream_frames")
                self._report_fps_if_due()
            except Exception as exc:
                self.source_stat_inc(
                    "network_stream_errors",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                print(
                    f"[DirectSbsStream] output failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self.shutdown_event.set()
                return


class _PyNvDirectSbsOutputMixin:
    """Encode CUDA video with PyNvVideoCodec and mux optional PCM audio."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pynv_output: PyNvSrtVideoOutput | None = None
        self._pynv_encoder: PyNvVideoCodecEncoder | None = None
        self._fallback_output: FfmpegDirectSbsOutput | None = None

    def _start_pynv_audio(self) -> str | None:
        if not self.stereo_mix_device:
            return None
        if self.os_name != "Windows" or not self.stereo_mix_device.casefold().startswith(
            "soundcard:"
        ):
            raise RuntimeError(
                "PyNvVideoCodec audio mux currently requires Windows soundcard loopback"
            )
        device_name = self.stereo_mix_device.split(":", 1)[1].strip()
        self._soundcard_audio = SoundcardLoopbackSender(device_name or None)
        self._soundcard_audio.start()
        return self._soundcard_audio.ffmpeg_url

    def _pynv_output_args(self) -> list[str]:
        if self.protocol == "WEBRTC":
            return [
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                "-pkt_size",
                "1452",
                f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                "?pkt_size=1452",
            ]
        return [
            "-f",
            "mpegts",
            "-mpegts_flags",
            "+resend_headers",
            f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316",
        ]

    def _release_pynv_pipeline(self) -> None:
        if self._pynv_output is not None:
            try:
                self._pynv_output.close()
            except Exception:
                pass
            self._pynv_output = None
        self._pynv_encoder = None
        if self._soundcard_audio is not None:
            self._soundcard_audio.close()
            self._soundcard_audio = None

    def _start_ffmpeg(self, width: int, height: int) -> None:
        nvc = _load_pynvvideo_codec()
        if nvc is None:
            raise RuntimeError(f"PyNvVideoCodec unavailable: {_PYNVVIDEO_CODEC_ERROR}")
        audio_url = self._start_pynv_audio()
        codec = "hevc" if self.use_hevc else "h264"
        self._pynv_encoder = PyNvVideoCodecEncoder(
            nvc,
            width,
            height,
            hevc=self.use_hevc,
            fps=self.fps,
            bitrate=max(
                1,
                int(
                    (self._dynamic_stream_rate_budget(width, height) or (10,))[0]
                    * 1_000_000
                ),
            ),
        )
        self._pynv_output = PyNvSrtVideoOutput(
            self._pynv_encoder,
            str(self.ffmpeg_path),
            codec=codec,
            fps=self.fps,
            audio_url=audio_url,
            audio_delay=self.audio_delay,
            audio_codec="libopus" if self.protocol == "WEBRTC" else "aac",
            output_args=self._pynv_output_args(),
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if self.os_name == "Windows"
                else 0
            ),
        )
        time.sleep(0.05)
        if self._pynv_output.process.poll() is not None:
            raise RuntimeError(
                "PyNvVideoCodec FFmpeg muxer exited during startup with code "
                f"{self._pynv_output.process.returncode}"
            )
        self._frame_size = (width, height)
        audio_codec_label = "Opus" if self.protocol == "WEBRTC" else "AAC"
        audio_label = f" + SoundCard/{audio_codec_label}" if audio_url else ""
        print(
            f"[DirectSbsStream] PyNvVideoCodec {codec} GPU path active"
            f"{audio_label}: {width}x{height}@{self.fps}",
            flush=True,
        )

    def _fallback_to_ffmpeg(self, frame: Any, reason: Exception) -> None:
        print(
            f"[DirectSbsStream] PyNvVideoCodec runtime failure: {reason}; "
            "falling back to FFmpeg video/audio encoding",
            flush=True,
        )
        self._release_pynv_pipeline()
        self._fallback_output = FfmpegDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
        )
        self._fallback_output.server_process = self.server_process
        self._fallback_output.submit_frame(runtime_sbs_to_rgb(frame))

    def submit_cuda_frame(self, frame: Any) -> None:
        if self._fallback_output is not None:
            self._fallback_output.submit_frame(runtime_sbs_to_rgb(frame))
            return
        try:
            if self._pynv_output is None:
                height, width = int(frame.shape[-2]), int(frame.shape[-1])
                if int(frame.shape[0]) not in (1, 3, 4):
                    height, width = int(frame.shape[0]), int(frame.shape[1])
                self._start_ffmpeg(width, height)
            assert self._pynv_output is not None
            self._pynv_output.submit_cuda_frame(frame)
        except Exception as exc:
            self._fallback_to_ffmpeg(frame, exc)

    def close(self) -> None:
        self._release_pynv_pipeline()
        if self._fallback_output is not None:
            self._fallback_output.close()
            self._fallback_output = None
        super().close()


class MjpegDirectSbsOutput:
    def __init__(self, *, port: int, fps: int, quality: int) -> None:
        profile = EncoderProfile(
            codec="mjpeg",
            quality=quality,
            target_fps=fps,
            pixel_format="rgb",
        )
        self.streamer = MJPEGStreamer(port=int(port), profile=profile)

    def start(self) -> None:
        self.streamer.start()
        print("[DirectSbsStream] MJPEG consumes packed SBS frames directly", flush=True)

    def submit_frame(self, frame: np.ndarray) -> None:
        self.streamer.set_frame(frame)

    def close(self) -> None:
        self.streamer.stop()


class FfmpegDirectSbsOutput:
    """Publish RGB SBS frames to MediaMTX through FFmpeg rawvideo stdin."""

    synchronous_submit = True

    def __init__(
        self,
        *,
        base_dir: str | Path,
        protocol: str,
        port: int,
        stream_key: str,
        fps: int,
        crf: int,
        stereo_mix_device: str | None = None,
        audio_delay: float = -0.1,
        os_name: str | None = None,
        prefer_nvenc: bool = False,
        display_mode: str = "Half-SBS",
        target_bitrate_mbps: int = 0,
        peak_bitrate_mbps: int = 0,
        auto_calibration: bool = False,
        calibration_port: int | None = None,
        on_calibration_fps: Callable[[int], Any] | None = None,
        calibration_fingerprint: dict[str, str] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.protocol = str(protocol or "RTMP").strip().upper()
        self.port = max(1, int(port))
        self.stream_key = str(stream_key or "live").strip() or "live"
        self.fps = max(1, int(fps))
        self.requested_fps = self.fps
        self.crf = max(0, min(51, int(crf)))
        self.stereo_mix_device = str(stereo_mix_device or "").strip()
        self.audio_delay = float(audio_delay)
        self._soundcard_audio: SoundcardLoopbackSender | None = None
        self.os_name = str(os_name or platform.system())
        self.prefer_nvenc = bool(prefer_nvenc)
        self.display_mode = str(display_mode or "Half-SBS").strip()
        self.use_hevc = self.display_mode.casefold() == "full-sbs"
        self.target_bitrate_mbps = max(0, int(target_bitrate_mbps))
        self.peak_bitrate_mbps = max(0, int(peak_bitrate_mbps))
        self.auto_calibration = bool(auto_calibration and self.protocol == "WEBRTC")
        self.calibration_port = int(calibration_port or min(65535, self.port + 1))
        self._on_calibration_fps = on_calibration_fps
        self._calibration_controller: StreamCalibrationController | None = None
        self.video_encoder = "libx265" if self.use_hevc else "libx264"
        self._active_rate_budget: tuple[int, int, int] | None = None
        self._encoder_selected = False
        runtime_root = Path(
            os.environ.get(
                "D2S_STREAMING_RUNTIME_DIR",
                self.base_dir / "streaming" / "rtmp",
            )
        )
        ffmpeg_name = "ffmpeg.exe" if self.os_name == "Windows" else "ffmpeg"
        mediamtx_name = "mediamtx.exe" if self.os_name == "Windows" else "mediamtx"
        if (runtime_root / "runtime-manifest.json").is_file():
            ensure_runtime(runtime_root)
        self.ffmpeg_path = self._find_executable(
            "D2S_FFMPEG_PATH",
            runtime_root / "ffmpeg" / "bin" / ffmpeg_name,
            "ffmpeg",
        )
        self.mediamtx_path = self._find_executable(
            "D2S_MEDIAMTX_PATH",
            runtime_root / "mediamtx" / mediamtx_name,
            "mediamtx",
        )
        self.mediamtx_config = Path(
            os.environ.get(
                "D2S_MEDIAMTX_CONFIG",
                runtime_root / "mediamtx.yml",
            )
        )
        if not self.mediamtx_config.is_file():
            raise FileNotFoundError(f"MediaMTX config not found: {self.mediamtx_config}")
        self.server_process: subprocess.Popen | None = None
        self.ffmpeg_process: subprocess.Popen | None = None
        self._ffmpeg_log_thread: threading.Thread | None = None
        self._ffmpeg_stderr_tail: list[str] = []
        self._ffmpeg_bitrate_mbps = 0.0
        self._mediamtx_inbound_bitrate_mbps = 0.0
        self._mediamtx_metrics_stop = threading.Event()
        self._mediamtx_metrics_thread: threading.Thread | None = None
        self._mediamtx_metrics_address = "127.0.0.1:9998"
        self._server_log_thread: threading.Thread | None = None
        self._frame_size: tuple[int, int] | None = None
        self._rate_probe_started: float | None = None
        self._rate_window_started: float | None = None
        self._rate_window_frames = 0
        self._rate_window_fps: list[float] = []
        self._rate_probe_min_seconds = 5.0
        self._rate_probe_max_seconds = 15.0
        self._stream_rate_calibrated = False
        self._next_submit_at = 0.0
        self._pending_audio_delay: float | None = None
        self._audio_delay_lock = threading.Lock()
        self._packet_loss_warning_emitted = False
        if self.auto_calibration:
            logs_dir = self.base_dir / "logs"
            self._calibration_controller = StreamCalibrationController(
                bind_port=self.calibration_port,
                stream_port=self.port,
                stream_key=self.stream_key,
                maximum_fps=self.requested_fps,
                state_path=logs_dir / "stream_calibration_state.json",
                profile_path=logs_dir / "stream_calibration_profile.json",
                hevc=self.use_hevc,
                fingerprint=calibration_fingerprint,
            )
            self._stream_rate_calibrated = True

    def prepare_calibration_source(self, runtime_result: Any) -> bool:
        """Start or retune the independent probe without converting RGB frames."""
        if self._calibration_controller is None:
            return False
        display_size = getattr(runtime_result, "output_display_size", None)
        if not display_size:
            frame = getattr(runtime_result, "sbs", runtime_result)
            shape = tuple(int(value) for value in getattr(frame, "shape", ()))
            if len(shape) == 4:
                display_size = (shape[-1], shape[-2])
            elif len(shape) == 3 and shape[0] in {1, 3, 4}:
                display_size = (shape[-1], shape[-2])
            elif len(shape) == 3:
                display_size = (shape[1], shape[0])
            else:
                raise ValueError("Unable to determine calibration stream resolution")
        output_width, output_height = (int(display_size[0]), int(display_size[1]))
        input_width = output_width
        if self.display_mode.casefold() == "full-sbs":
            input_width = max(1, output_width // 2)
        self._calibration_controller.configure_input_resolution(
            input_width, output_height
        )
        self._apply_pending_calibration_tier()
        size = (output_width, output_height)
        if self.ffmpeg_process is None:
            self._start_ffmpeg(*size)
        elif self._frame_size != size:
            raise RuntimeError(
                f"Calibration stream size changed from {self._frame_size} to {size}"
            )
        return True

    @staticmethod
    def _find_executable(env_name: str, bundled: Path, command: str) -> Path:
        configured = os.environ.get(env_name)
        candidates = [Path(configured)] if configured else []
        candidates.append(bundled)
        discovered = shutil.which(command)
        if discovered:
            candidates.append(Path(discovered))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"{command} not found; set {env_name} or install it under {bundled.parent}"
        )

    def _server_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self._calibration_controller is not None:
            env["MTX_METRICS"] = "yes"
            env["MTX_METRICSADDRESS"] = self._mediamtx_metrics_address
        if self.protocol == "RTMP":
            env["MTX_RTMPADDRESS"] = f":{self.port}"
        elif self.protocol == "RTSP":
            env["MTX_RTSPADDRESS"] = f":{self.port}"
        elif self.protocol in {"HLS", "HLS M3U8"}:
            env["MTX_HLSADDRESS"] = f":{self.port}"
        elif self.protocol == "WEBRTC":
            env["MTX_WEBRTCADDRESS"] = f":{self.port}"
        return env

    @staticmethod
    def _parse_mediamtx_path_inbound_bytes(metrics: str, path: str) -> int | None:
        for line in str(metrics or "").splitlines():
            if not line.startswith(("paths_inbound_bytes{", "paths_bytes_received{")):
                continue
            labels, separator, value = line.partition("}")
            if not separator or f'name="{path}"' not in labels:
                continue
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
        return None

    def _read_mediamtx_path_inbound_bytes(self) -> int | None:
        try:
            with urlopen(
                f"http://{self._mediamtx_metrics_address}/metrics"
                f"?type=paths&path={quote(self.stream_key)}",
                timeout=0.5,
            ) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        return self._parse_mediamtx_path_inbound_bytes(payload, self.stream_key)

    def _sample_mediamtx_metrics(self) -> None:
        previous_bytes: int | None = None
        previous_time: float | None = None
        while not self._mediamtx_metrics_stop.wait(1.0):
            current_bytes = self._read_mediamtx_path_inbound_bytes()
            current_time = time.monotonic()
            if current_bytes is None:
                continue
            if (
                previous_bytes is not None
                and previous_time is not None
                and current_bytes >= previous_bytes
            ):
                elapsed = current_time - previous_time
                if elapsed > 0:
                    self._mediamtx_inbound_bitrate_mbps = (
                        (current_bytes - previous_bytes) * 8.0 / elapsed / 1_000_000.0
                    )
            previous_bytes = current_bytes
            previous_time = current_time

    @property
    def publish_rtsp_port(self) -> int:
        return self.port if self.protocol == "RTSP" else 8554

    def _probe_encoder(self, encoder: str, width: int, height: int) -> bool:
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        command = [
            str(self.ffmpeg_path), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={int(width)}x{int(height)}:r=1",
            "-frames:v", "1", "-an",
        ]
        if encoder.endswith("_vaapi"):
            vaapi_device = os.environ.get("D2S_VAAPI_DEVICE", "/dev/dri/renderD128")
            command[1:1] = ["-vaapi_device", vaapi_device]
            command.extend(["-vf", "format=nv12,hwupload"])
        command.extend([
            "-c:v",
            encoder,
            "-pix_fmt",
            "nv12" if encoder.endswith("_vaapi") else "yuv420p",
            "-f",
            "null",
            "-",
        ])
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=8.0,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"[DirectSbsStream] {encoder} probe failed for {int(width)}x{int(height)}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        if result.returncode == 0:
            return True
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail_lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
        detail = detail_lines[-1] if detail_lines else f"FFmpeg exited with code {result.returncode}"
        if encoder.endswith("_nvenc"):
            print(
                f"[DirectSbsStream] NVENC probe failed for {int(width)}x{int(height)}: {detail}",
                flush=True,
            )
        else:
            print(f"[DirectSbsStream] {encoder} unavailable: {detail}", flush=True)
        return False

    def _probe_nvenc(self, width: int, height: int) -> bool:
        if self.os_name == "Darwin":
            return False
        return self._probe_encoder(
            "hevc_nvenc" if self.use_hevc else "h264_nvenc", width, height
        )

    def _encoder_candidates(self) -> list[tuple[str, str]]:
        codec = "hevc" if self.use_hevc else "h264"
        candidates: list[tuple[str, str]] = []
        if self.os_name == "Darwin":
            candidates.append((f"{codec}_videotoolbox", "Apple VideoToolbox"))
        else:
            if self.os_name == "Windows":
                candidates.append((f"{codec}_nvenc", "NVIDIA NVENC"))
                candidates.extend([
                    (f"{codec}_qsv", "Intel Quick Sync"),
                    (f"{codec}_amf", "AMD AMF"),
                ])
            elif self.os_name == "Linux":
                candidates.extend([
                    (f"{codec}_qsv", "Intel Quick Sync"),
                    (f"{codec}_vaapi", "VAAPI"),
                ])
        software_encoder = "libx265" if self.use_hevc else "libx264"
        candidates.append((software_encoder, "software"))
        return candidates

    def _select_video_encoder(self, width: int, height: int) -> str:
        codec_label = "H.265/HEVC" if self.use_hevc else "H.264"
        candidates = self._encoder_candidates()
        for encoder, label in candidates[:-1]:
            supported = (
                self._probe_nvenc(width, height)
                if encoder.endswith("_nvenc")
                else self._probe_encoder(encoder, width, height)
            )
            if supported:
                self._encoder_selection_reason = label
                print(
                    f"[DirectSbsStream] {label} {codec_label} encoder active: {encoder}",
                    flush=True,
                )
                return encoder
        software_encoder, _ = candidates[-1]
        self._encoder_selection_reason = "software fallback"
        print(
            f"[DirectSbsStream] {codec_label} hardware encoders unavailable; "
            f"falling back to {software_encoder}",
            flush=True,
        )
        return software_encoder

    def _dynamic_stream_rate_budget(
        self, width: int, height: int
    ) -> tuple[int, int, int] | None:
        """Return wireless-friendly target, peak and VBV rates in Mbps."""
        if self.protocol not in {"HLS", "HLS M3U8", "RTMP", "WEBRTC"}:
            return None
        configured_target = int(getattr(self, "target_bitrate_mbps", 0) or 0)
        if configured_target > 0:
            configured_peak = int(getattr(self, "peak_bitrate_mbps", 0) or 0)
            peak = max(configured_target, configured_peak)
            return configured_target, peak, peak
        pixels_per_second = max(1, int(width)) * max(1, int(height)) * self.fps
        bits_per_pixel = 0.075 if self.use_hevc else 0.12
        quality_factor = max(0.5, min(2.0, 2.0 ** ((20 - self.crf) / 12.0)))
        target_limit = 100 if self.use_hevc else 80
        peak_limit = 120 if self.use_hevc else 100
        target_mbps = round(
            pixels_per_second * bits_per_pixel * quality_factor / 1_000_000
        )
        target_mbps = max(4, min(target_limit, target_mbps))
        peak_mbps = max(
            target_mbps,
            min(peak_limit, int(math.ceil(target_mbps * 1.15))),
        )
        return target_mbps, peak_mbps, peak_mbps

    @staticmethod
    def _select_sustainable_stream_fps(
        measured_fps: float, maximum_fps: int
    ) -> int:
        maximum = max(1, int(maximum_fps))
        safe_limit = min(float(maximum), max(1.0, float(measured_fps)) * 0.90)
        for candidate in (60, 50, 48, 40, 30, 25, 24, 20, 15, 12, 10):
            if candidate <= maximum and candidate <= safe_limit:
                return candidate
        return max(5, min(maximum, int(safe_limit)))

    @staticmethod
    def _stable_rate_sample(window_fps: list[float]) -> float | None:
        recent = window_fps[-5:]
        if len(recent) < 5:
            return None
        median_fps = float(statistics.median(recent))
        if statistics.pstdev(recent) > max(1.0, median_fps * 0.06):
            return None
        ordered = sorted(recent)
        return float(ordered[max(0, int((len(ordered) - 1) * 0.20))])

    @staticmethod
    def _fallback_rate_sample(window_fps: list[float]) -> float:
        ordered = sorted(window_fps[-10:])
        if not ordered:
            return 1.0
        return float(ordered[max(0, int((len(ordered) - 1) * 0.20))])

    def should_submit_frame(self, now: float | None = None) -> bool:
        self._apply_pending_calibration_tier()
        timestamp = time.perf_counter() if now is None else float(now)
        if not self._stream_rate_calibrated:
            if self._rate_probe_started is None:
                self._rate_probe_started = timestamp
                self._rate_window_started = timestamp
                self._rate_window_frames = 1
                return False

            self._rate_window_frames += 1
            window_elapsed = timestamp - float(self._rate_window_started)
            if window_elapsed >= 1.0:
                self._rate_window_fps.append(
                    self._rate_window_frames / window_elapsed
                )
                self._rate_window_started = timestamp
                self._rate_window_frames = 0

            elapsed = timestamp - self._rate_probe_started
            stable_fps = self._stable_rate_sample(self._rate_window_fps)
            if elapsed < self._rate_probe_min_seconds or (
                stable_fps is None and elapsed < self._rate_probe_max_seconds
            ):
                return False
            measured_fps = (
                stable_fps
                if stable_fps is not None
                else self._fallback_rate_sample(self._rate_window_fps)
            )
            self.fps = self._select_sustainable_stream_fps(
                measured_fps, self.requested_fps
            )
            self._stream_rate_calibrated = True
            self._next_submit_at = timestamp + 1.0 / float(self.fps)
            print(
                f"[DirectSbsStream] Stable stream rate selected: "
                f"measured={measured_fps:.1f} target={self.fps} FPS "
                f"windows={len(self._rate_window_fps)}",
                flush=True,
            )
            return True

        interval = 1.0 / float(self.fps)
        if timestamp + 1e-9 < self._next_submit_at:
            return False
        if timestamp - self._next_submit_at > interval:
            self._next_submit_at = timestamp + interval
        else:
            self._next_submit_at += interval
        return True

    def _apply_pending_calibration_tier(self) -> None:
        controller = getattr(self, "_calibration_controller", None)
        if controller is None:
            return
        tier = controller.take_pending_tier()
        if tier is None:
            return
        changed = (
            self.fps != tier.fps
            or self.target_bitrate_mbps != tier.target_mbps
            or self.peak_bitrate_mbps != tier.peak_mbps
        )
        self.fps = tier.fps
        self.target_bitrate_mbps = tier.target_mbps
        self.peak_bitrate_mbps = tier.peak_mbps
        self._next_submit_at = 0.0
        if callable(self._on_calibration_fps):
            self._on_calibration_fps(tier.fps)
        if changed and self.ffmpeg_process is not None:
            print(
                f"[StreamCalibration] Testing {tier.fps} FPS "
                f"target={tier.target_mbps}M peak={tier.peak_mbps}M",
                flush=True,
            )
            self._stop_process(self.ffmpeg_process)
            self.ffmpeg_process = None
            self._frame_size = None

    def observe_calibration_window(
        self,
        *,
        sbs_fps: float,
        submitted_fps: float,
        convert_ms: float,
        submit_ms: float,
    ) -> None:
        if getattr(self, "_calibration_controller", None) is None:
            return
        measured_bitrate = float(
            self._mediamtx_inbound_bitrate_mbps
            or self._ffmpeg_bitrate_mbps
            or 0.0
        )
        self._calibration_controller.observe_sender(
            {
                "sbs_fps": round(float(sbs_fps), 3),
                "submitted_fps": (
                    30.0
                    if self._calibration_controller is not None
                    else round(float(submitted_fps), 3)
                ),
                "convert_ms": round(float(convert_ms), 3),
                "submit_ms": round(float(submit_ms), 3),
                "encoded_bitrate_mbps": round(measured_bitrate, 3),
            }
        )

    @staticmethod
    def _looks_like_packet_loss(message: str) -> bool:
        normalized = str(message or "").casefold()
        indicators = (
            "packet loss",
            "packet lost",
            "packet missed",
            "missing packet",
            "dropped packet",
            "rtp packet gap",
            "buffer underflow",
        )
        return any(indicator in normalized for indicator in indicators)

    def _drain_mediamtx_output(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            message = str(line).rstrip("\r\n")
            if not message:
                continue
            print(f"[MediaMTX] {message}", flush=True)
            if self._looks_like_packet_loss(message) and not self._packet_loss_warning_emitted:
                self._packet_loss_warning_emitted = True
                print(
                    "[DirectSbsStream] WARNING: MediaMTX reported possible UDP packet loss; "
                    "consider increasing udpReadBufferSize (currently 0) and check network/CPU load.",
                    flush=True,
                )

    def start(self) -> None:
        if self.protocol != "WEBRTC":
            print(
                f"[DirectSbsStream] WARNING: {self.protocol} selected; "
                "WebRTC is recommended for lower-latency browser streaming.",
                flush=True,
            )
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        self.server_process = subprocess.Popen(
            [str(self.mediamtx_path), str(self.mediamtx_config)],
            cwd=str(self.mediamtx_config.parent),
            env=self._server_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        time.sleep(0.25)
        if self.server_process.poll() is not None:
            startup_output = ""
            try:
                startup_output, _ = self.server_process.communicate(timeout=1.0)
            except (OSError, subprocess.SubprocessError):
                pass
            output_lines = [
                line.strip() for line in startup_output.splitlines() if line.strip()
            ]
            detail = next(
                (line for line in reversed(output_lines) if " ERR " in line),
                output_lines[-1] if output_lines else "unknown startup error",
            )
            raise RuntimeError(f"MediaMTX exited during startup: {detail}")
        self._server_log_thread = threading.Thread(
            target=self._drain_mediamtx_output,
            args=(self.server_process,),
            name="MediaMTXLog",
            daemon=True,
        )
        self._server_log_thread.start()
        if self._calibration_controller is not None:
            self._mediamtx_metrics_stop.clear()
            self._mediamtx_metrics_thread = threading.Thread(
                target=self._sample_mediamtx_metrics,
                name="MediaMTXMetrics",
                daemon=True,
            )
            self._mediamtx_metrics_thread.start()
            self._calibration_controller.start()
        print(
            f"[DirectSbsStream] MediaMTX started for {self.protocol} on port {self.port}",
            flush=True,
        )

    def _audio_input_args(self) -> list[str]:
        device = self.stereo_mix_device
        if not device or device.lower().startswith(("no ", "none", "null")):
            return []
        if self.os_name == "Windows":
            if device.casefold().startswith("soundcard:"):
                if self._soundcard_audio is None:
                    return []
                return [
                    "-itsoffset",
                    str(self.audio_delay),
                    "-f",
                    "s16le",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-i",
                    self._soundcard_audio.ffmpeg_url,
                ]
            if device.casefold().startswith("wasapi:"):
                return [
                    "-itsoffset",
                    str(self.audio_delay),
                    "-f",
                    "wasapi",
                    "-i",
                    device.split(":", 1)[1].strip(),
                ]
            return [
                "-itsoffset",
                str(self.audio_delay),
                "-f",
                "dshow",
                "-i",
                f"audio={device}",
            ]
        if self.os_name == "Linux":
            return [
                "-itsoffset",
                str(self.audio_delay),
                "-f",
                "pulse",
                "-i",
                device,
            ]
        if self.os_name == "Darwin":
            audio_device = device
            if audio_device.isdigit():
                audio_device = f":{audio_device}"
            elif not audio_device.startswith(":"):
                audio_device = f":{audio_device}"
            return [
                "-itsoffset",
                str(self.audio_delay),
                "-f",
                "avfoundation",
                "-i",
                audio_device,
            ]
        return []

    def _ffmpeg_command(self, width: int, height: int) -> list[str]:
        calibration_stream = getattr(self, "_calibration_controller", None) is not None
        # Calibration must not inherit inference/RGB24 throughput. FFmpeg owns
        # the clock and produces a deterministic 30 FPS pressure stream.
        audio_args = [] if calibration_stream else self._audio_input_args()
        self._active_rate_budget = self._dynamic_stream_rate_budget(width, height)
        target_rate = (
            f"{self._active_rate_budget[0]}M"
            if self._active_rate_budget is not None
            else "0"
        )
        input_args = (
            [
                "-re",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={width}x{height}:rate={self.fps}",
            ]
            if calibration_stream
            else [
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:0",
            ]
        )
        command = [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "64",
            "-analyzeduration",
            "0",
            *input_args,
            *audio_args,
            "-map",
            "0:v:0",
        ]
        if audio_args:
            command.extend(["-map", "1:a:0"])
        if self.video_encoder in {"h264_videotoolbox", "hevc_videotoolbox"}:
            command.extend(
                [
                    "-c:v",
                    self.video_encoder,
                    "-profile:v",
                    "main" if self.use_hevc else "high",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    target_rate if self._active_rate_budget is not None else "10M",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-realtime",
                    "true",
                ]
            )
        elif self.video_encoder in {
            "h264_nvenc", "hevc_nvenc",
            "h264_qsv", "hevc_qsv",
            "h264_amf", "hevc_amf",
            "h264_vaapi", "hevc_vaapi",
        }:
            command.extend(
                [
                    "-c:v",
                    self.video_encoder,
                    "-preset",
                    "p1" if self.video_encoder.endswith("_nvenc") else "fast",
                    "-tune",
                    "ll" if self.video_encoder.endswith(("_nvenc", "_qsv")) else "zerolatency",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(self.crf),
                    "-b:v",
                    target_rate,
                    *(
                        ["-vf", "format=nv12,hwupload"]
                        if self.video_encoder.endswith("_vaapi")
                        else []
                    ),
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-zerolatency",
                    "1",
                    "-forced-idr",
                    "1",
                    "-strict_gop",
                    "1",
                    "-spatial-aq",
                    "1",
                    "-temporal-aq",
                    "1",
                    "-aq-strength",
                    "8",
                ]
            )
        elif self.video_encoder == "libx265":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-crf",
                    str(self.crf),
                    "-x265-params",
                    f"keyint={self.fps}:min-keyint={self.fps}:scenecut=0:"
                    "rc-lookahead=0:open-gop=0:repeat-headers=1",
                ]
            )
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-crf",
                    str(self.crf),
                    "-x264-params",
                    f"keyint={self.fps}:min-keyint={self.fps}:scenecut=0:"
                    "rc-lookahead=0:open-gop=0:repeat-headers=1",
                ]
            )
        if self.video_encoder.endswith("_vaapi"):
            vaapi_device = os.environ.get("D2S_VAAPI_DEVICE", "/dev/dri/renderD128")
            command[1:1] = ["-vaapi_device", vaapi_device]
            pix_fmt_index = command.index("-pix_fmt")
            command[pix_fmt_index + 1] = "nv12"
            for option in ("-tune", "-rc", "-cq", "-zerolatency", "-forced-idr", "-strict_gop", "-spatial-aq", "-temporal-aq", "-aq-strength"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
        elif self.video_encoder.endswith("_qsv"):
            for option in ("-tune", "-rc", "-cq", "-zerolatency", "-forced-idr", "-strict_gop", "-spatial-aq", "-temporal-aq", "-aq-strength"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
            command.extend(["-global_quality", str(self.crf), "-look_ahead", "0"])
        elif self.video_encoder.endswith("_amf"):
            for option in ("-tune", "-rc", "-cq", "-zerolatency", "-forced-idr", "-strict_gop", "-spatial-aq", "-temporal-aq", "-aq-strength"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
            command.extend(["-usage", "ultralowlatency", "-quality", "speed", "-rc", "vbr_peak"])

        if calibration_stream:
            # Normal playback remains quality-oriented VBR. Calibration uses
            # constant-rate output so each tier applies the requested load to
            # the PC-to-headset network instead of merely changing a VBR cap.
            for option in ("-cq", "-crf", "-global_quality"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
            if self.video_encoder.endswith("_nvenc"):
                rc_index = command.index("-rc")
                command[rc_index + 1] = "cbr"
            elif self.video_encoder.endswith("_amf"):
                rc_index = command.index("-rc")
                command[rc_index + 1] = "cbr"
            if "-b:v" not in command:
                command.extend(["-b:v", target_rate])
            command.extend(["-minrate", target_rate])

        if self._active_rate_budget is not None:
            target_mbps, peak_mbps, buffer_mbps = self._active_rate_budget
            if calibration_stream:
                peak_mbps = target_mbps
                buffer_mbps = target_mbps
            command.extend(
                [
                    "-maxrate",
                    f"{peak_mbps}M",
                    "-bufsize",
                    f"{buffer_mbps}M",
                ]
            )
        if audio_args:
            if self.protocol == "WEBRTC":
                command.extend(
                    [
                        "-af",
                        "aresample=async=1000:first_pts=0",
                        "-c:a",
                        "libopus",
                        "-ar",
                        "48000",
                        "-ac",
                        "2",
                        "-b:a",
                        "96k",
                    ]
                )
            else:
                command.extend(["-c:a", "aac", "-ar", "48000", "-b:a", "128k"])
        if getattr(self, "_calibration_controller", None) is not None:
            # FFmpeg reports the actual encoded/muxed output rate, which is
            # different from the configured target bitrate for VBR content.
            command.extend(["-progress", "pipe:2", "-stats_period", "1"])
        if self.os_name == "Windows":
            command.extend(
                [
                    "-force_key_frames",
                    "expr:gte(t,n_forced*1)",
                    "-muxdelay",
                    "0",
                    "-muxpreload",
                    "0",
                    "-flush_packets",
                    "1",
                    # Zero waits indefinitely for every stream and can starve RTSP.
                    "-max_interleave_delta",
                    "100000",
                ]
            )
            if self.protocol == "WEBRTC":
                # Publish WebRTC's source through the local RTSP listener.
                # This avoids waiting for SRT MPEG-TS stream discovery while
                # MediaMTX still performs the RTSP→WebRTC conversion.
                command.extend(
                    [
                        "-f",
                        "rtsp",
                        "-rtsp_transport",
                        "tcp",
                        "-pkt_size",
                        "1452",
                        f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                        "?pkt_size=1452",
                    ]
                )
            else:
                command.extend(
                    [
                        "-f",
                        "mpegts",
                        "-mpegts_flags",
                        "+resend_headers",
                        f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316",
                    ]
                )
        else:
            command.extend(
                [
                    "-threads",
                    "2",
                    # Bound sparse-stream interleaving to 100 ms.
                    "-max_interleave_delta",
                    "100000",
                    "-f",
                    "rtsp",
                    "-rtsp_transport",
                    "tcp",
                    "-pkt_size",
                    "1452",
                    f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                    "?pkt_size=1452",
                ]
            )
        return command

    def _start_ffmpeg(self, width: int, height: int) -> None:
        if (
            self._calibration_controller is None
            and self.os_name == "Windows"
            and self.stereo_mix_device.casefold().startswith("soundcard:")
        ):
            try:
                device_name = self.stereo_mix_device.split(":", 1)[1].strip()
                self._soundcard_audio = SoundcardLoopbackSender(device_name or None)
                self._soundcard_audio.start()
            except Exception as exc:
                print(
                    f"[DirectSbsStream] soundcard loopback unavailable: {exc}; "
                    "falling back to dshow",
                    flush=True,
                )
                self._soundcard_audio = None
                self.stereo_mix_device = device_name
        if not self._encoder_selected:
            self.video_encoder = self._select_video_encoder(width, height)
            self._encoder_selected = True
        command = self._ffmpeg_command(width, height)
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=(subprocess.DEVNULL if self._calibration_controller is not None else subprocess.PIPE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._ffmpeg_stderr_tail = []
        self._ffmpeg_log_thread = threading.Thread(
            target=self._drain_ffmpeg_stderr,
            args=(self.ffmpeg_process,),
            name="DirectSbsFfmpegLog",
            daemon=True,
        )
        self._ffmpeg_log_thread.start()
        time.sleep(0.05)
        if self.ffmpeg_process.poll() is not None:
            detail = "; ".join(self._ffmpeg_stderr_tail[-3:]) or "no FFmpeg diagnostic"
            raise RuntimeError(
                f"FFmpeg exited during startup with code {self.ffmpeg_process.returncode}: {detail}"
            )
        self._frame_size = (width, height)
        if self._active_rate_budget is not None:
            target_mbps, peak_mbps, buffer_mbps = self._active_rate_budget
            print(
                f"[DirectSbsStream] Dynamic stream quality: protocol={self.protocol} "
                f"target={target_mbps}M "
                f"peak={peak_mbps}M buffer={buffer_mbps}M "
                f"resolution={width}x{height} fps={self.fps} crf={self.crf}",
                flush=True,
            )
        if self._calibration_controller is not None:
            print(
                f"[StreamCalibration] Independent CBR pressure stream active: "
                f"{width}x{height}@{self.fps} encoder={self.video_encoder}",
                flush=True,
            )
        else:
            print(
                f"[DirectSbsStream] FFmpeg consumes RGB24 SBS directly: "
                f"{width}x{height}@{self.fps} encoder={self.video_encoder}",
                flush=True,
            )

    def _write_frame(self, frame: np.ndarray) -> None:
        process = self.ffmpeg_process
        if process is None or process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        if process.poll() is not None:
            detail = "; ".join(self._ffmpeg_stderr_tail[-3:]) or "no FFmpeg diagnostic"
            raise RuntimeError(
                f"FFmpeg exited with code {process.returncode}: {detail}"
            )
        process.stdin.write(memoryview(frame).cast("B"))
        process.stdin.flush()

    def _drain_ffmpeg_stderr(self, process: subprocess.Popen) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for raw_line in stream:
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw_line).strip()
                if not line:
                    continue
                match = re.match(
                    r"bitrate=\s*([0-9.]+)([kmg]?bits/s)",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).casefold()
                    multiplier = {
                        "bits/s": 1e-6,
                        "kbits/s": 1e-3,
                        "mbits/s": 1.0,
                    }[unit]
                    self._ffmpeg_bitrate_mbps = value * multiplier
                self._ffmpeg_stderr_tail.append(line)
                del self._ffmpeg_stderr_tail[:-20]
                if any(token in line.casefold() for token in ("error", "failed", "invalid", "cannot")):
                    print(f"[DirectSbsStream] FFmpeg: {line}", flush=True)
        except (OSError, ValueError):
            return

    def request_audio_delay(self, delay: float) -> bool:
        delay = max(-10.0, min(10.0, float(delay)))
        with self._audio_delay_lock:
            current = (
                self._pending_audio_delay
                if self._pending_audio_delay is not None
                else self.audio_delay
            )
            if math.isclose(delay, current, abs_tol=1e-6):
                return False
            self._pending_audio_delay = delay
        return True

    def _apply_pending_audio_delay(self) -> None:
        with self._audio_delay_lock:
            delay = self._pending_audio_delay
            self._pending_audio_delay = None
        if delay is None or math.isclose(delay, self.audio_delay, abs_tol=1e-6):
            return
        previous = self.audio_delay
        self.audio_delay = delay
        if self.ffmpeg_process is None:
            return
        print(
            f"[DirectSbsStream] Audio delay changed: {previous:.3f}s -> "
            f"{delay:.3f}s; restarting FFmpeg publisher",
            flush=True,
        )
        self._stop_process(self.ffmpeg_process)
        self.ffmpeg_process = None
        self._frame_size = None

    def submit_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        size = (int(width), int(height))
        self._apply_pending_audio_delay()
        if self.ffmpeg_process is None:
            self._start_ffmpeg(*size)
        elif self._frame_size != size:
            raise RuntimeError(
                f"SBS stream size changed from {self._frame_size} to {size}; restart required"
            )
        if self._calibration_controller is not None:
            # The independent lavfi source runs continuously at 30 FPS. Runtime
            # frames are intentionally ignored so RGB conversion/inference
            # speed cannot throttle the bandwidth probe.
            return
        try:
            self._write_frame(frame)
        except (BrokenPipeError, OSError, RuntimeError):
            if self.video_encoder not in {"h264_nvenc", "hevc_nvenc"}:
                raise
            software_encoder = "libx265" if self.use_hevc else "libx264"
            print(
                f"[DirectSbsStream] NVENC startup failed; retrying with "
                f"{software_encoder}",
                flush=True,
            )
            self._stop_process(self.ffmpeg_process)
            self.ffmpeg_process = None
            self._frame_size = None
            self.video_encoder = software_encoder
            self._start_ffmpeg(*size)
            self._write_frame(frame)

    @staticmethod
    def _stop_process(process: subprocess.Popen | None) -> None:
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)

    def close(self) -> None:
        self._mediamtx_metrics_stop.set()
        if self._mediamtx_metrics_thread is not None:
            self._mediamtx_metrics_thread.join(timeout=1.5)
            self._mediamtx_metrics_thread = None
        if self._calibration_controller is not None:
            self._calibration_controller.close()
            self._calibration_controller = None
        if self._soundcard_audio is not None:
            self._soundcard_audio.close()
            self._soundcard_audio = None
        self._stop_process(self.ffmpeg_process)
        self._stop_process(self.server_process)
        if self._server_log_thread is not None:
            self._server_log_thread.join(timeout=0.5)
        self.ffmpeg_process = None
        self.server_process = None
        self._server_log_thread = None
        self._ffmpeg_log_thread = None
        self._ffmpeg_stderr_tail = []


class PyNvDirectSbsOutput(_PyNvDirectSbsOutputMixin, FfmpegDirectSbsOutput):
    pass


class _AmdAmfDirectSbsOutputMixin:
    """Submit ROCm tensors to the native HIP→D3D11→AMF bridge."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._amd_encoder = None
        self._amd_packet_process: subprocess.Popen | None = None
        self._amd_fallback: FfmpegDirectSbsOutput | None = None

    @staticmethod
    def _hip_rgba_tensor(frame):
        import torch

        image = getattr(frame, "sbs", frame)
        if not isinstance(image, torch.Tensor) or not bool(getattr(image, "is_cuda", False)):
            raise RuntimeError("AMD AMF requires a ROCm device tensor")
        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise RuntimeError("AMD AMF accepts one SBS frame at a time")
            image = image[0]
        if image.ndim != 3:
            raise RuntimeError(f"unsupported AMD AMF tensor shape: {tuple(image.shape)!r}")
        if int(image.shape[0]) in (3, 4):
            image = image.permute(1, 2, 0)
        if int(image.shape[-1]) not in (3, 4):
            raise RuntimeError(f"AMD AMF requires RGB/RGBA tensor: {tuple(image.shape)!r}")
        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        if int(image.shape[-1]) == 3:
            alpha = torch.full(
                (*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device
            )
            image = torch.cat((image, alpha), dim=-1)
        return image.contiguous()

    def _start_amd_encoder(self, image) -> None:
        if self.stereo_mix_device:
            raise RuntimeError("native AMD AMF path requires audio to be disabled")
        from streaming.amd_encoder import AmdAmfSurfaceEncoder

        height, width = int(image.shape[0]), int(image.shape[1])
        budget = self._dynamic_stream_rate_budget(width, height)
        bitrate = int((budget[0] if budget is not None else 10) * 1_000_000)
        self._amd_encoder = AmdAmfSurfaceEncoder(
            width, height, self.fps, bitrate, hevc=self.use_hevc
        )
        codec = "hevc" if self.use_hevc else "h264"
        destination = (
            f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316"
        )
        self._amd_packet_process = subprocess.Popen(
            [
                str(self.ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "nobuffer",
                "-f",
                codec,
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "copy",
                "-f",
                "mpegts",
                destination,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if self.os_name == "Windows"
                else 0
            ),
        )
        self._frame_size = (width, height)
        print(
            f"[DirectSbsStream] AMD HIP→D3D11→AMF GPU path active: "
            f"{width}x{height}@{self.fps}",
            flush=True,
        )

    def _submit_amd_packet(self, image) -> None:
        import torch

        if self._amd_encoder is None or self._amd_packet_process is None:
            self._start_amd_encoder(image)
        assert self._amd_encoder is not None
        process = self._amd_packet_process
        if process.poll() is not None or process.stdin is None:
            raise RuntimeError("AMF packet muxer is unavailable")
        stream = int(torch.cuda.current_stream(device=image.device).cuda_stream)
        self._amd_encoder.submit_hip_rgba(
            int(image.data_ptr()), int(image.stride(0) * image.element_size()), stream
        )
        while True:
            packet = self._amd_encoder.read_packet()
            if not packet:
                break
            process.stdin.write(packet)
            process.stdin.flush()

    def _fallback_to_ffmpeg(self, frame, reason: Exception) -> None:
        print(
            f"[DirectSbsStream] AMD native GPU path unavailable: {reason}; "
            "falling back to FFmpeg hardware/software encoding",
            flush=True,
        )
        if self._amd_packet_process is not None:
            self._stop_process(self._amd_packet_process)
            self._amd_packet_process = None
        if self._amd_encoder is not None:
            self._amd_encoder.close()
            self._amd_encoder = None
        self._amd_fallback = FfmpegDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
        )
        self._amd_fallback.server_process = self.server_process
        self._amd_fallback.submit_frame(runtime_sbs_to_rgb(frame))

    def submit_cuda_frame(self, frame: Any) -> None:
        if self._amd_fallback is not None:
            self._amd_fallback.submit_frame(runtime_sbs_to_rgb(frame))
            return
        try:
            self._submit_amd_packet(self._hip_rgba_tensor(frame))
        except Exception as exc:
            self._fallback_to_ffmpeg(frame, exc)

    def close(self) -> None:
        if self._amd_packet_process is not None:
            self._stop_process(self._amd_packet_process)
            self._amd_packet_process = None
        if self._amd_encoder is not None:
            self._amd_encoder.close()
            self._amd_encoder = None
        if self._amd_fallback is not None:
            self._amd_fallback.close()
            self._amd_fallback = None
        super().close()


class AmdAmfDirectSbsOutput(_AmdAmfDirectSbsOutputMixin, FfmpegDirectSbsOutput):
    pass
