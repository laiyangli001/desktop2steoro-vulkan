from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderProfile:
    codec: str = "mjpeg"
    quality: int = 85
    target_fps: int = 30
    target_bitrate: int | None = None
    resize_width: int | None = None
    resize_height: int | None = None
    pixel_format: str = "rgb"

    def __post_init__(self):
        codec = str(self.codec or "mjpeg").strip().lower()
        pixel_format = str(self.pixel_format or "rgb").strip().lower()
        object.__setattr__(self, "codec", codec)
        object.__setattr__(self, "pixel_format", pixel_format)
        object.__setattr__(self, "quality", _clamp_int(self.quality, 1, 100, 85))
        object.__setattr__(self, "target_fps", _clamp_int(self.target_fps, 1, 240, 30))
        object.__setattr__(self, "resize_width", _optional_positive_int(self.resize_width))
        object.__setattr__(self, "resize_height", _optional_positive_int(self.resize_height))
        if codec not in {"mjpeg", "h264", "h265"}:
            raise ValueError(f"Unsupported encoder codec: {self.codec}")
        if pixel_format not in {"rgb", "bgra", "bgr", "nv12"}:
            raise ValueError(f"Unsupported encoder pixel format: {self.pixel_format}")

    @property
    def resize_size(self) -> tuple[int, int] | None:
        if self.resize_width is None or self.resize_height is None:
            return None
        return self.resize_width, self.resize_height


def encoder_profile_from_stream_settings(settings: dict) -> EncoderProfile:
    return EncoderProfile(
        codec=str(settings.get("Stream Codec", "mjpeg")),
        quality=settings.get("Stream Quality", 85),
        target_fps=settings.get("Target FPS", settings.get("Capture FPS", 30)) or 30,
        target_bitrate=settings.get("Stream Bitrate"),
        resize_width=settings.get("Stream Resize Width"),
        resize_height=settings.get("Stream Resize Height"),
        pixel_format=str(settings.get("Stream Pixel Format", "rgb")),
    )


def _clamp_int(value, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_positive_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
