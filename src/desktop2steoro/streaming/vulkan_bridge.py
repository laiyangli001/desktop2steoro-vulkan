"""Optional native AV_PIX_FMT_VULKAN bridge loader.

The host-upload Vulkan path must remain available when the native bridge is
not installed. This module only accepts the versioned narrow ABI; it never
silently treats a random DLL as a zero-copy encoder.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


EXPECTED_ABI_VERSION = 1
REQUIRED_SYMBOLS = (
    "d2s_vulkan_ffmpeg_bridge_abi_version",
    "d2s_vulkan_ffmpeg_bridge_probe",
    "d2s_vulkan_ffmpeg_encoder_create",
    "d2s_vulkan_ffmpeg_encoder_submit_image",
    "d2s_vulkan_ffmpeg_encoder_read_packet",
    "d2s_vulkan_ffmpeg_encoder_destroy",
)


class VulkanNativeBridge:
    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        missing = [name for name in REQUIRED_SYMBOLS if not hasattr(library, name)]
        if missing:
            raise RuntimeError("missing ABI symbols: " + ", ".join(missing))
        version = int(library.d2s_vulkan_ffmpeg_bridge_abi_version())
        if version != EXPECTED_ABI_VERSION:
            raise RuntimeError(f"unsupported ABI version: {version}")
        probe = library.d2s_vulkan_ffmpeg_bridge_probe
        probe.argtypes = [ctypes.c_char_p, ctypes.c_int]
        probe.restype = ctypes.c_int
        message = ctypes.create_string_buffer(512)
        if int(probe(message, len(message))) != 1:
            detail = message.value.decode("utf-8", errors="replace")
            raise RuntimeError(detail or "native Vulkan FFmpeg probe failed")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "VulkanNativeBridge":
        configured = path or os.environ.get("D2S_VULKAN_FFMPEG_BRIDGE")
        if not configured:
            raise FileNotFoundError("D2S_VULKAN_FFMPEG_BRIDGE is not configured")
        candidate = Path(configured)
        if not candidate.is_file():
            raise FileNotFoundError(f"Vulkan FFmpeg bridge not found: {candidate}")
        return cls(ctypes.CDLL(str(candidate)))
