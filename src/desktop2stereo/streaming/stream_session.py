"""Shared policy and configuration for network streaming sessions."""

from __future__ import annotations

from dataclasses import dataclass


NETWORK_STREAM_MODES = frozenset({"RTMP Streamer", "GPU Streamer"})
CALIBRATABLE_STREAM_MODES = NETWORK_STREAM_MODES


def is_network_stream_mode(run_mode: str) -> bool:
    return str(run_mode or "").strip() in NETWORK_STREAM_MODES


def supports_network_calibration(run_mode: str, protocol: str) -> bool:
    return (
        str(run_mode or "").strip() in CALIBRATABLE_STREAM_MODES
        and str(protocol or "").strip().casefold() == "webrtc"
    )


@dataclass(frozen=True)
class NetworkStreamSessionConfig:
    """Transport settings shared by vendor and advanced video backends."""

    protocol: str
    port: int
    stream_key: str
    fps: int
    crf: int
    stereo_mix_device: str
    audio_delay: float
    display_mode: str
    target_bitrate_mbps: int = 0
    peak_bitrate_mbps: int = 0

    @classmethod
    def from_settings(cls, settings: dict, *, fps: int) -> "NetworkStreamSessionConfig":
        return cls(
            protocol=str(settings.get("Stream Protocol", "WebRTC") or "WebRTC"),
            port=int(settings.get("Streamer Port", 1122)),
            stream_key=str(settings.get("Stream Key", "live") or "live"),
            fps=int(fps),
            crf=int(settings.get("CRF", 23)),
            stereo_mix_device=str(settings.get("Stereo Mix", "") or ""),
            audio_delay=float(settings.get("Audio Delay", -0.1)),
            display_mode=str(settings.get("Display Mode", "Half-SBS") or "Half-SBS"),
            target_bitrate_mbps=int(settings.get("Stream Target Bitrate Mbps", 0) or 0),
            peak_bitrate_mbps=int(settings.get("Stream Peak Bitrate Mbps", 0) or 0),
        )
