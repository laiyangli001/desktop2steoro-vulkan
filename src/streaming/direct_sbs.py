from __future__ import annotations

import os
from pathlib import Path
import platform
import queue
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np

from streaming.encoder_profile import EncoderProfile
from streaming.mjpeg_streamer import MJPEGStreamer


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
        fps_report_interval: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.runtime_q = runtime_q
        self.shutdown_event = shutdown_event
        self.output = output
        self.source_stat_inc = source_stat_inc
        self.show_fps_provider = show_fps_provider
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
        show_fps = (
            bool(self.show_fps_provider())
            if self.show_fps_provider is not None
            else False
        )
        if show_fps:
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
                should_submit = getattr(self.output, "should_submit_frame", None)
                if callable(should_submit) and not should_submit(self._clock()):
                    self._report_fps_if_due()
                    continue
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
        audio_delay: float = 0.0,
        os_name: str | None = None,
        prefer_nvenc: bool = False,
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
        self.os_name = str(os_name or platform.system())
        self.prefer_nvenc = bool(prefer_nvenc)
        self.video_encoder = "libx264"
        self._encoder_selected = False
        self.ffmpeg_path = self._find_executable(
            "D2S_FFMPEG_PATH",
            self.base_dir / "streaming" / "rtmp" / "ffmpeg" / "bin" / (
                "ffmpeg.exe" if self.os_name == "Windows" else "ffmpeg"
            ),
            "ffmpeg",
        )
        self.mediamtx_path = self._find_executable(
            "D2S_MEDIAMTX_PATH",
            self.base_dir / "streaming" / "rtmp" / "mediamtx" / (
                "mediamtx.exe" if self.os_name == "Windows" else "mediamtx"
            ),
            "mediamtx",
        )
        self.mediamtx_config = self.mediamtx_path.with_name("mediamtx.yml")
        if not self.mediamtx_config.is_file():
            raise FileNotFoundError(f"MediaMTX config not found: {self.mediamtx_config}")
        self.server_process: subprocess.Popen | None = None
        self.ffmpeg_process: subprocess.Popen | None = None
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
        if self.protocol == "RTMP":
            env["MTX_RTMPADDRESS"] = f":{self.port}"
        elif self.protocol == "RTSP":
            env["MTX_RTSPADDRESS"] = f":{self.port}"
        elif self.protocol in {"HLS", "HLS M3U8"}:
            env["MTX_HLSADDRESS"] = f":{self.port}"
        elif self.protocol == "WEBRTC":
            env["MTX_WEBRTCADDRESS"] = f":{self.port}"
        return env

    @property
    def publish_rtsp_port(self) -> int:
        return self.port if self.protocol == "RTSP" else 8554

    def _probe_nvenc(self, width: int, height: int) -> bool:
        if self.os_name == "Darwin":
            return False
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        try:
            result = subprocess.run(
                [
                    str(self.ffmpeg_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={int(width)}x{int(height)}:r=1",
                    "-frames:v",
                    "1",
                    "-an",
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p1",
                    "-tune",
                    "ll",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(self.crf),
                    "-b:v",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-zerolatency",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=8.0,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _select_video_encoder(self, width: int, height: int) -> str:
        if self.prefer_nvenc and self._probe_nvenc(width, height):
            print(
                "[DirectSbsStream] NVIDIA NVENC H.264 encoder active",
                flush=True,
            )
            return "h264_nvenc"
        if self.prefer_nvenc:
            print(
                "[DirectSbsStream] NVENC unavailable; falling back to libx264",
                flush=True,
            )
        return "libx264"

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

    @staticmethod
    def _drain_mediamtx_output(process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            message = str(line).rstrip("\r\n")
            if message:
                print(f"[MediaMTX] {message}", flush=True)

    def start(self) -> None:
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
        print(
            f"[DirectSbsStream] MediaMTX started for {self.protocol} on port {self.port}",
            flush=True,
        )

    def _audio_input_args(self) -> list[str]:
        device = self.stereo_mix_device
        if not device or device.lower().startswith(("no ", "none", "null")):
            return []
        if self.os_name == "Windows":
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
        return []

    def _ffmpeg_command(self, width: int, height: int) -> list[str]:
        audio_args = self._audio_input_args()
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
            *audio_args,
            "-map",
            "0:v:0",
        ]
        if audio_args:
            command.extend(["-map", "1:a:0"])
        if self.os_name == "Darwin":
            command.extend(
                [
                    "-c:v",
                    "h264_videotoolbox",
                    "-profile:v",
                    "high",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    "10M",
                    "-maxrate",
                    "12M",
                    "-bufsize",
                    "24M",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-realtime",
                    "true",
                ]
            )
        elif self.video_encoder == "h264_nvenc":
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p1",
                    "-tune",
                    "ll",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(self.crf),
                    "-b:v",
                    "0",
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
                ]
            )
            if self.os_name == "Linux":
                command.extend(
                    [
                        "-x264-params",
                        f"keyint={self.fps}:min-keyint={self.fps}:scenecut=0:rc-lookahead=0",
                    ]
                )
        if audio_args:
            if self.os_name == "Windows":
                command.extend(["-c:a", "libopus", "-b:a", "96k"])
            else:
                command.extend(["-c:a", "aac", "-ar", "44100", "-b:a", "96k"])
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
                    "-f",
                    "mpegts",
                    f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316",
                ]
            )
        else:
            command.extend(
                [
                    "-threads",
                    "2",
                    "-f",
                    "rtsp",
                    "-rtsp_transport",
                    "tcp",
                    f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}",
                ]
            )
        return command

    def _start_ffmpeg(self, width: int, height: int) -> None:
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
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self._frame_size = (width, height)
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
            raise RuntimeError(f"FFmpeg exited with code {process.returncode}")
        process.stdin.write(memoryview(frame).cast("B"))

    def submit_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        size = (int(width), int(height))
        if self.ffmpeg_process is None:
            self._start_ffmpeg(*size)
        elif self._frame_size != size:
            raise RuntimeError(
                f"SBS stream size changed from {self._frame_size} to {size}; restart required"
            )
        try:
            self._write_frame(frame)
        except (BrokenPipeError, OSError, RuntimeError):
            if self.video_encoder != "h264_nvenc":
                raise
            print(
                "[DirectSbsStream] NVENC startup failed; retrying with libx264",
                flush=True,
            )
            self._stop_process(self.ffmpeg_process)
            self.ffmpeg_process = None
            self._frame_size = None
            self.video_encoder = "libx264"
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
        self._stop_process(self.ffmpeg_process)
        self._stop_process(self.server_process)
        if self._server_log_thread is not None:
            self._server_log_thread.join(timeout=0.5)
        self.ffmpeg_process = None
        self.server_process = None
        self._server_log_thread = None
