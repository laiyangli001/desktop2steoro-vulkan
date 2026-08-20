"""Optional Windows AMD AMF bridge loader.

The native bridge is intentionally optional. It only reports whether the AMD
AMF runtime and an AMD DXGI adapter are available; surface encoding is enabled
by the follow-up AMF/D3D11 frame path and never falls back silently to a CPU
"zero-copy" claim.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        Path(__file__).resolve().with_name("amd_encoder") / "d2s_amd_encoder.dll",
        root / "native" / "windows" / "d2s_amd_encoder.dll",
        root / "native" / "amd_encoder" / "d2s_amd_encoder.dll",
        Path(os.environ.get("D2S_AMD_ENCODER_DLL", "")),
    ]


def probe_amd_amf() -> tuple[bool, str]:
    """Return (available, diagnostic) without importing any GPU Python package."""

    if os.name != "nt":
        return False, "AMD AMF bridge is Windows-only"
    for candidate in _library_candidates():
        if not candidate or not candidate.exists():
            continue
        try:
            bridge = ctypes.WinDLL(str(candidate))
            bridge.d2s_amd_encoder_probe.restype = ctypes.c_int
            bridge.d2s_amd_encoder_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
            bridge.d2s_amd_encoder_last_error.restype = ctypes.c_int
            if bridge.d2s_amd_encoder_probe():
                return True, "AMD AMF runtime and DXGI adapter detected"
            buffer = ctypes.create_string_buffer(512)
            bridge.d2s_amd_encoder_last_error(buffer, len(buffer))
            return False, buffer.value.decode("utf-8", errors="replace")
        except OSError as exc:
            return False, f"AMD bridge load failed: {exc}"
    return False, "d2s_amd_encoder.dll is not installed"


class AmdAmfSurfaceEncoder:
    """Thin wrapper for the native D3D11-surface AMF encoder.

    The texture argument must be an existing ``ID3D11Texture2D`` pointer. The
    wrapper deliberately does not accept NumPy or CPU buffers: callers that
    cannot provide a GPU surface must use the normal FFmpeg fallback.
    """

    def __init__(self, width: int, height: int, fps: int, bitrate: int, *, hevc: bool = False):
        available, detail = probe_amd_amf()
        if not available:
            raise RuntimeError(detail)
        dll = next(path for path in _library_candidates() if path and path.exists())
        self._dll = ctypes.WinDLL(str(dll))
        self._dll.d2s_amd_encoder_create.restype = ctypes.c_void_p
        self._dll.d2s_amd_encoder_create.argtypes = [ctypes.c_int] * 4 + [ctypes.c_int]
        self._dll.d2s_amd_encoder_submit_texture.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._dll.d2s_amd_encoder_submit_texture.restype = ctypes.c_int
        self._dll.d2s_amd_encoder_read_packet.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._dll.d2s_amd_encoder_read_packet.restype = ctypes.c_int
        self._dll.d2s_amd_encoder_destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._dll.d2s_amd_encoder_create(width, height, fps, bitrate, int(hevc))
        if not self._handle:
            raise RuntimeError("AMF surface encoder initialization failed")

    def submit_d3d11_texture(self, texture_pointer: int) -> None:
        if self._dll.d2s_amd_encoder_submit_texture(self._handle, ctypes.c_void_p(texture_pointer)) <= 0:
            raise RuntimeError("AMF rejected the D3D11 texture")

    def read_packet(self, capacity: int = 2 * 1024 * 1024) -> bytes | None:
        buffer = ctypes.create_string_buffer(capacity)
        size = self._dll.d2s_amd_encoder_read_packet(self._handle, buffer, capacity)
        if size <= 0:
            return None
        return buffer.raw[:size]

    def close(self) -> None:
        if self._handle:
            self._dll.d2s_amd_encoder_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
