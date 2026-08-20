from __future__ import annotations

from dataclasses import dataclass
import subprocess
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
    rgb = image.float()
    if float(rgb.detach().amax().item()) <= 1.0:
        rgb = rgb * 255.0
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

    def __init__(self, nvc: Any, width: int, height: int, *, hevc: bool, fps: int, bitrate: int):
        codec = "hevc" if hevc else "h264"
        self._encoder = nvc.CreateEncoder(
            int(width), int(height), "NV12", False,
            codec=codec, fps=int(fps), bitrate=int(bitrate), gpu_id=0,
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
    """Encode CUDA frames and remux H.264/HEVC packets to MPEG-TS/SRT."""

    def __init__(self, encoder: PyNvVideoCodecEncoder, ffmpeg_path: str, srt_url: str, *, codec: str = "h264"):
        self.encoder = encoder
        self.process = subprocess.Popen(
            [
                str(ffmpeg_path), "-hide_banner", "-loglevel", "warning",
                "-f", "hevc" if codec.casefold() in {"hevc", "h265"} else "h264",
                "-i", "pipe:0", "-c:v", "copy", "-f", "mpegts", srt_url,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def submit_cuda_frame(self, frame: Any) -> None:
        if self.process.stdin is None:
            raise RuntimeError("SRT remuxer stdin is unavailable")
        packet = self.encoder.encode(frame)
        if packet:
            self.process.stdin.write(packet)
            self.process.stdin.flush()

    def close(self) -> None:
        try:
            tail = self.encoder.flush()
            if tail and self.process.stdin is not None:
                self.process.stdin.write(tail)
                self.process.stdin.flush()
        finally:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=5.0)
