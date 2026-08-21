"""Runtime checks for FFmpeg Vulkan Video encoding.

Presence of ``h264_vulkan`` in ``ffmpeg -encoders`` only proves that the
encoder was compiled.  The driver must also expose a usable Vulkan Video
profile and accept the input format used by the stream.  This module keeps
that distinction explicit so a failed probe can trigger one safe fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class VulkanVideoProbeReport:
    encoder: str
    width: int
    height: int
    available: bool
    encoder_compiled: bool
    device_initialized: bool
    input_format: str = "nv12"
    detail: str = ""


def _creationflags(os_name: str) -> int:
    return (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if os_name == "Windows"
        else 0
    )


def _run(
    command: Sequence[str], *, timeout: float, os_name: str
) -> tuple[int, str]:
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1.0, float(timeout)),
            creationflags=_creationflags(os_name),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return int(result.returncode), str(result.stdout or "")


def probe_vulkan_video(
    ffmpeg_path: str | Path,
    *,
    width: int,
    height: int,
    hevc: bool = False,
    timeout: float = 8.0,
    os_name: str | None = None,
) -> VulkanVideoProbeReport:
    """Run the same real-device smoke test required by the build guide."""

    encoder = "hevc_vulkan" if hevc else "h264_vulkan"
    system = str(os_name or platform.system())
    ffmpeg = str(Path(ffmpeg_path))
    code, listing = _run(
        [ffmpeg, "-hide_banner", "-encoders"],
        timeout=timeout,
        os_name=system,
    )
    compiled = code == 0 and encoder in listing
    if not compiled:
        return VulkanVideoProbeReport(
            encoder, int(width), int(height), False, False, False,
            detail=f"FFmpeg encoder is not available: {encoder}",
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-init_hw_device",
        "vulkan=d2s_vk:0",
        "-filter_hw_device",
        "d2s_vk",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={int(width)}x{int(height)}:rate=1",
        "-vf",
        "format=nv12,hwupload",
        "-frames:v",
        "1",
        "-c:v",
        encoder,
        "-profile:v",
        "main" if hevc else "high",
        *(["-level:v", "5.1"] if not hevc and int(width) >= 2560 else []),
        "-b:v",
        "2M",
        "-f",
        "null",
        "-",
    ]
    probe_code, output = _run(command, timeout=timeout, os_name=system)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = lines[-1] if lines else f"FFmpeg exited with code {probe_code}"
    return VulkanVideoProbeReport(
        encoder,
        int(width),
        int(height),
        probe_code == 0,
        True,
        probe_code == 0,
        detail=detail,
    )
