"""Optional oneVPL D3D11 NV12-surface encoder bridge.

The bridge accepts only a borrowed ID3D11Texture2D. It never accepts NumPy or
host buffers, and it is unavailable unless built with the oneVPL SDK.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from .native_artifacts import native_dll_candidates


_DLL_NAME = "d2s_onevpl_d3d11_encoder.dll"


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[5]
    return native_dll_candidates(
        _DLL_NAME,
        environment_variable="D2S_ONEVPL_D3D11_DLL",
        extra_directories=(
            root / "native" / "onevpl_d3d11_encoder",
            root / "src" / "desktop2stereo" / "stereo_runtime" / "providers" / "intel" / "native" / "onevpl_d3d11_encoder",
        ),
    )


def _load_bridge():
    if os.name != "nt":
        return None
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            library = ctypes.WinDLL(str(path))
        except OSError:
            continue
        required = (
            "d2s_onevpl_d3d11_probe",
            "d2s_onevpl_d3d11_last_error",
            "d2s_onevpl_d3d11_create",
            "d2s_onevpl_d3d11_adapter_luid",
            "d2s_onevpl_d3d11_submit_nv12",
            "d2s_onevpl_d3d11_read_packet",
            "d2s_onevpl_d3d11_destroy",
        )
        if any(not hasattr(library, name) for name in required):
            continue
        library.d2s_onevpl_d3d11_probe.argtypes = []
        library.d2s_onevpl_d3d11_probe.restype = ctypes.c_int
        library.d2s_onevpl_d3d11_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.d2s_onevpl_d3d11_last_error.restype = ctypes.c_int
        library.d2s_onevpl_d3d11_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        library.d2s_onevpl_d3d11_create.restype = ctypes.c_void_p
        library.d2s_onevpl_d3d11_adapter_luid.argtypes = [ctypes.c_void_p]
        library.d2s_onevpl_d3d11_adapter_luid.restype = ctypes.c_ulonglong
        library.d2s_onevpl_d3d11_submit_nv12.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_longlong,
        ]
        library.d2s_onevpl_d3d11_submit_nv12.restype = ctypes.c_int
        library.d2s_onevpl_d3d11_read_packet.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.d2s_onevpl_d3d11_read_packet.restype = ctypes.c_int
        library.d2s_onevpl_d3d11_destroy.argtypes = [ctypes.c_void_p]
        library.d2s_onevpl_d3d11_destroy.restype = None
        return library
    return None


def _last_error(library) -> str:
    buffer = ctypes.create_string_buffer(1024)
    library.d2s_onevpl_d3d11_last_error(buffer, len(buffer))
    return buffer.value.decode("utf-8", errors="replace") or "oneVPL bridge operation failed"


def probe_onevpl_d3d11() -> dict[str, object]:
    library = _load_bridge()
    if library is None:
        return {
            "available": False,
            "backend": "onevpl_d3d11_surface",
            "reason": "oneVPL D3D11 encoder bridge DLL is not installed",
        }
    if not library.d2s_onevpl_d3d11_probe():
        return {
            "available": False,
            "backend": "onevpl_d3d11_surface",
            "reason": _last_error(library),
        }
    return {
        "available": True,
        "backend": "onevpl_d3d11_surface",
        "input": "borrowed_nv12_d3d11_texture",
        "gpu_to_cpu": False,
        "zero_copy": True,
        "reason": None,
    }


class OneVPLD3D11SurfaceEncoder:
    """Encode borrowed NV12 D3D11 textures using oneVPL."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        d3d11_device: int,
        hevc: bool = False,
    ) -> None:
        self._library = _load_bridge()
        if self._library is None:
            raise RuntimeError("oneVPL D3D11 encoder bridge DLL is unavailable")
        device = ctypes.c_void_p(d3d11_device)
        self._handle = self._library.d2s_onevpl_d3d11_create(
            int(width),
            int(height),
            int(fps),
            int(bitrate),
            int(bool(hevc)),
            device,
        )
        if not self._handle:
            raise RuntimeError(_last_error(self._library))

    def submit_nv12(self, texture: int, timestamp: int) -> None:
        result = self._library.d2s_onevpl_d3d11_submit_nv12(
            self._handle,
            ctypes.c_void_p(int(texture)),
            int(timestamp),
        )
        if result != 1:
            raise RuntimeError(_last_error(self._library))

    @property
    def adapter_luid(self) -> int:
        return int(self._library.d2s_onevpl_d3d11_adapter_luid(self._handle))

    def read_packet(self, capacity: int = 4 * 1024 * 1024) -> bytes | None:
        output = ctypes.create_string_buffer(int(capacity))
        result = self._library.d2s_onevpl_d3d11_read_packet(
            self._handle, output, int(capacity)
        )
        if result < 0:
            raise RuntimeError(_last_error(self._library))
        if result == 0:
            return None
        return output.raw[:result]

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.d2s_onevpl_d3d11_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "OneVPLD3D11SurfaceEncoder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["OneVPLD3D11SurfaceEncoder", "probe_onevpl_d3d11"]
