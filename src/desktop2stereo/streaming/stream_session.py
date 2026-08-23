"""Shared policy and configuration for network streaming sessions."""

from __future__ import annotations

from dataclasses import dataclass


NETWORK_STREAM_MODES = frozenset({"RTMP Streamer", "GPU Streamer"})
CALIBRATABLE_STREAM_MODES = NETWORK_STREAM_MODES


@dataclass(frozen=True)
class NetworkVideoBackendDecision:
    """Resolved video backend shared by every network-streaming GUI mode."""

    backend: str
    requested: str
    reason: str


def resolve_network_video_backend(
    run_mode: str,
    requested_backend: str,
    *,
    device_info: str,
) -> NetworkVideoBackendDecision:
    """Resolve the encoder once for RTMP and GPU streaming modes.

    The two GUI modes intentionally keep their labels and compatibility
    behavior, but now share the same backend policy.  Explicit choices always
    win; ``auto`` selects the vendor zero-copy path for GPU Streamer and leaves
    Advanced Network Streaming on the FFmpeg/MediaMTX path.
    """

    requested = str(requested_backend or "auto").strip().casefold()
    if requested in {"", "automatic"}:
        requested = "auto"
    if requested == "qsv":
        requested = "intel"
    if requested not in {"auto", "ffmpeg", "pynv", "amd", "intel", "vulkan"}:
        requested = "auto"

    if requested != "auto":
        return NetworkVideoBackendDecision(
            backend=requested,
            requested=requested,
            reason="explicit GUI encoder selection",
        )

    info = str(device_info or "").upper()
    if str(run_mode or "").strip() == "GPU Streamer":
        if "NVIDIA" in info:
            return NetworkVideoBackendDecision("pynv", requested, "NVIDIA GPU auto selection")
        if "AMD" in info or "RADEON" in info:
            return NetworkVideoBackendDecision("amd", requested, "AMD GPU auto selection")
        if "INTEL" in info:
            return NetworkVideoBackendDecision("intel", requested, "Intel zero-copy auto selection")
    return NetworkVideoBackendDecision("ffmpeg", requested, "shared FFmpeg/MediaMTX fallback")


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
