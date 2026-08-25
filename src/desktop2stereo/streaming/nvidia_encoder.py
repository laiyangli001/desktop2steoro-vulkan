from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import subprocess
import threading
from typing import Any


def rgb_cuda_to_nv12(frame: Any) -> tuple[Any, Any]:
    """Convert a CUDA RGB tensor to NV12 planes without downloading to CPU."""
    import torch

    image = frame.detach()
    if not bool(getattr(image, "is_cuda", False)):
        raise ValueError("PyNvVideoCodec GPU input requires a CUDA tensor")
    if image.ndim == 4:
        if int(image.shape[0]) != 1:
            raise ValueError(f"expected one frame, got {tuple(image.shape)!r}")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected RGB HWC or CHW tensor, got {tuple(image.shape)!r}")
    if int(image.shape[0]) in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    elif int(image.shape[-1]) in (1, 3, 4):
        image = image[..., :3]
    else:
        raise ValueError(f"unsupported RGB tensor shape: {tuple(image.shape)!r}")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if height % 2 or width % 2:
        raise ValueError("NV12 requires even width and height")
    if image.dtype == torch.uint8:
        rgb = image.float()
    else:
        # Runtime floating-point SBS output is normalized to 0..1.
        rgb = image.float().clamp(0.0, 1.0) * 255.0
    r, g, b = rgb.unbind(dim=-1)
    y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp(0, 255).to(torch.uint8)
    u = (-0.148 * r - 0.291 * g + 0.439 * b + 128.0).clamp(0, 255)
    v = (0.439 * r - 0.368 * g - 0.071 * b + 128.0).clamp(0, 255)
    u = u.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    v = v.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    uv = torch.stack((u, v), dim=-1).reshape(height // 2, width).to(torch.uint8)
    nv12 = torch.empty((height * 3 // 2, width, 1), dtype=torch.uint8, device=image.device)
    y_plane = nv12[:height]
    uv_plane = nv12[height:].reshape(height // 2, width // 2, 2)
    y_plane.copy_(y.unsqueeze(-1))
    uv_plane.copy_(uv.reshape(height // 2, width // 2, 2))
    # PyNvVideoCodec owns a separate CUDA/NVENC submission queue and cannot
    # infer a dependency on PyTorch's current stream. Without this hand-off
    # wait it may read the NV12 planes while conversion is still in flight,
    # producing an initialized encoder with corrupted frames.
    torch.cuda.current_stream(device=image.device).synchronize()
    return y_plane, uv_plane


@dataclass
class Nv12CudaFrame:
    """CUDA Array Interface adapter expected by PyNvVideoCodec GPU input."""

    y: Any
    uv: Any

    def cuda(self) -> list[Any]:
        return [self.y, self.uv]


class PyNvVideoCodecEncoder:
    """Small encoder adapter; muxing and transport remain outside this class."""

    def __init__(
        self,
        nvc: Any,
        width: int,
        height: int,
        *,
        hevc: bool,
        fps: int,
        bitrate: int,
        gpu_id: int = 0,
    ):
        codec = "hevc" if hevc else "h264"
        frame_rate = max(1, int(fps))
        target_bitrate = max(1, int(bitrate))
        self._encoder = nvc.CreateEncoder(
            int(width),
            int(height),
            "NV12",
            False,
            codec=codec,
            fps=frame_rate,
            bitrate=target_bitrate,
            maxbitrate=max(target_bitrate, int(target_bitrate * 1.2)),
            gpu_id=max(0, int(gpu_id)),
            tuning_info="ultra_low_latency",
            preset="P1",
            rc="cbr",
            gop=frame_rate,
            idrperiod=frame_rate,
            # WebRTC low-latency streams must remain IPPP. B-frames require
            # display reordering and can make a slow browser reader lose the
            # reference chain when MediaMTX has to discard queued packets.
            bf=0,
            repeatspspps=1,
        )

    def encode(self, rgb_cuda: Any) -> bytes:
        y, uv = rgb_cuda_to_nv12(rgb_cuda)
        packet = self._encoder.Encode(Nv12CudaFrame(y, uv))
        return self._packet_bytes(packet)

    def flush(self) -> bytes:
        return self._packet_bytes(self._encoder.EndEncode())

    @staticmethod
    def _packet_bytes(packets: Any) -> bytes:
        if packets is None:
            return b""
        if isinstance(packets, (bytes, bytearray, memoryview)):
            return bytes(packets)
        output = bytearray()
        for packet in packets:
            if isinstance(packet, dict):
                packet = packet.get("data", b"")
            output.extend(bytes(packet))
        return bytes(output)


class PyNvSrtVideoOutput:
    """Mux PyNvVideoCodec packets with optional PCM audio through FFmpeg."""

    def __init__(
        self,
        encoder: Any,
        ffmpeg_path: str,
        srt_url: str | None = None,
        *,
        codec: str = "h264",
        fps: int = 30,
        audio_url: str | None = None,
        audio_delay: float = 0.0,
        audio_codec: str = "libopus",
        output_args: list[str] | None = None,
        creationflags: int = 0,
    ):
        self.encoder = encoder
        if output_args is None:
            if not srt_url:
                raise ValueError("srt_url or output_args is required")
            output_args = ["-f", "mpegts", srt_url]
        command = [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            # PyNv packets are Annex-B bytes without AVPacket PTS/DTS.
            # Generate a monotonic video timeline before RTSP interleaving
            # with the live Opus input.
            "-fflags",
            "+nobuffer+genpts",
            "-flags",
            "low_delay",
            "-probesize",
            "64",
            "-analyzeduration",
            "0",
            "-f",
            "hevc" if codec.casefold() in {"hevc", "h265"} else "h264",
            "-r",
            str(max(1, int(fps))),
            "-i",
            "pipe:0",
        ]
        if audio_url:
            command.extend(
                [
                    "-thread_queue_size",
                    "1024",
                    "-itsoffset",
                    str(float(audio_delay)),
                    "-f",
                    "s16le",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-i",
                    audio_url,
                ]
            )
        command.extend(["-map", "0:v:0"])
        if audio_url:
            command.extend(["-map", "1:a:0"])
        command.extend(["-c:v", "copy", "-fps_mode", "cfr"])
        if audio_url:
            if audio_codec == "libopus":
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
        else:
            command.append("-an")
        command.extend(
            [
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-flush_packets",
                "1",
                "-max_interleave_delta",
                "100000",
                *output_args,
            ]
        )
        self.command = command
        self._stderr_tail = deque(maxlen=40)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=int(creationflags),
        )
        self._stderr_thread = None
        stderr = getattr(self.process, "stderr", None)
        if stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(stderr,),
                name="NativeNvencMuxLog",
                daemon=True,
            )
            self._stderr_thread.start()
        print(
            "[DirectSbsStream] NativeNVENC mux active: "
            f"audio={'enabled' if audio_url else 'disabled'} "
            f"audio_codec={audio_codec if audio_url else 'none'} "
            f"audio_input={audio_url or 'none'}",
            flush=True,
        )

    def _drain_stderr(self, stderr) -> None:
        try:
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
                print(f"[DirectSbsStream] NativeNVENC mux: {line}", flush=True)
        except (OSError, ValueError):
            return

    def submit_cuda_frame(self, frame: Any) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"NVENC packet muxer exited with code {self.process.returncode}"
            )
        if self.process.stdin is None:
            raise RuntimeError("NVENC packet muxer stdin is unavailable")
        packet = self.encoder.encode(frame)
        if packet:
            self.process.stdin.write(packet)
            self.process.stdin.flush()

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                tail = self.encoder.flush()
                if tail and self.process.stdin is not None:
                    self.process.stdin.write(tail)
                    self.process.stdin.flush()
        finally:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=3.0)
            close_encoder = getattr(self.encoder, "close", None)
            if callable(close_encoder):
                close_encoder()
