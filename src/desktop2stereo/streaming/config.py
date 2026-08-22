from __future__ import annotations

from dataclasses import dataclass

from utils.network import get_local_ip


DEFAULT_PORT = 1122


@dataclass(frozen=True)
class StreamingConfig:
    stream_quality: int
    stream_port: int
    local_ip: str
    stereo_mix_device: str
    stream_key: str
    audio_delay: float
    crf: int


def resolve_streaming_config(settings: dict) -> StreamingConfig:
    return StreamingConfig(
        stream_quality=settings["Stream Quality"],
        stream_port=settings["Streamer Port"],
        local_ip=get_local_ip(),
        # Audio output is selected dynamically by SoundCard/WASAPI. Keep an
        # empty value when legacy settings do not contain a saved device.
        stereo_mix_device=settings.get("Stereo Mix", ""),
        stream_key=settings["Stream Key"],
        audio_delay=settings["Audio Delay"],
        crf=settings["CRF"],
    )
