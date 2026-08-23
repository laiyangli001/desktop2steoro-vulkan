"""Optional D3D11 surface owner for the final composed SBS frame.

The bridge intentionally accepts the final RGB frame only at this transitional
stage. It uploads into a D3D11 BGRA texture, converts it on the GPU to NV12,
and exposes the borrowed NV12 texture to a same-device oneVPL encoder.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from .native_artifacts import native_dll_candidates


_DLL_NAME = "d2s_d3d11_sbs_surface.dll"


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[5]
    return native_dll_candidates(
        _DLL_NAME,
        environment_variable="D2S_D3D11_SBS_SURFACE_DLL",
        extra_directories=(
            root / "native" / "d3d11_sbs_surface",
            root / "src" / "desktop2stereo" / "stereo_runtime" / "providers" / "intel" / "native" / "d3d11_sbs_surface",
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
            "d2s_d3d11_sbs_surface_probe",
            "d2s_d3d11_sbs_surface_create",
            "d2s_d3d11_sbs_surface_device",
            "d2s_d3d11_sbs_surface_adapter_luid",
            "d2s_d3d11_sbs_surface_upload_bgra",
            "d2s_d3d11_sbs_surface_nv12",
            "d2s_d3d11_sbs_surface_last_error",
            "d2s_d3d11_sbs_surface_destroy",
        )
        if any(not hasattr(library, name) for name in required):
            continue
        library.d2s_d3d11_sbs_surface_probe.argtypes = []
        library.d2s_d3d11_sbs_surface_probe.restype = ctypes.c_int
        library.d2s_d3d11_sbs_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        library.d2s_d3d11_sbs_surface_create.restype = ctypes.c_void_p
        library.d2s_d3d11_sbs_surface_device.argtypes = [ctypes.c_void_p]
        library.d2s_d3d11_sbs_surface_device.restype = ctypes.c_void_p
        library.d2s_d3d11_sbs_surface_adapter_luid.argtypes = [ctypes.c_void_p]
        library.d2s_d3d11_sbs_surface_adapter_luid.restype = ctypes.c_ulonglong
        library.d2s_d3d11_sbs_surface_upload_bgra.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        library.d2s_d3d11_sbs_surface_upload_bgra.restype = ctypes.c_int
        library.d2s_d3d11_sbs_surface_nv12.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.d2s_d3d11_sbs_surface_nv12.restype = ctypes.c_int
        library.d2s_d3d11_sbs_surface_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.d2s_d3d11_sbs_surface_last_error.restype = ctypes.c_int
        library.d2s_d3d11_sbs_surface_destroy.argtypes = [ctypes.c_void_p]
        library.d2s_d3d11_sbs_surface_destroy.restype = None
        return library
    return None


def _last_error(library) -> str:
    buffer = ctypes.create_string_buffer(1024)
    library.d2s_d3d11_sbs_surface_last_error(buffer, len(buffer))
    return buffer.value.decode("utf-8", errors="replace") or "D3D11 SBS surface operation failed"


def probe_d3d11_sbs_surface() -> dict[str, object]:
    library = _load_bridge()
    if library is None:
        return {
            "available": False,
            "backend": "d3d11_sbs_surface",
            "gpu_to_cpu": True,
            "zero_copy": False,
            "gpu_copy_count": 1,
            "reason": "D3D11 SBS surface bridge DLL is not installed",
        }
    capabilities = int(library.d2s_d3d11_sbs_surface_probe())
    return {
        "available": bool(capabilities & 0x03 == 0x03),
        "backend": "d3d11_sbs_surface",
        "input": "final_sbs_bgra8_cpu_upload",
        "output": "borrowed_nv12_d3d11_texture",
        "gpu_to_cpu": True,
        "zero_copy": False,
        "gpu_copy_count": 1,
        "reason": None if capabilities & 0x03 == 0x03 else "bridge capabilities are incomplete",
    }


class D3D11SbsSurface:
    """Own a D3D11 final-SBS conversion surface."""

    def __init__(self, *, width: int, height: int, adapter_index: int = -1) -> None:
        self._library = _load_bridge()
        if self._library is None:
            raise RuntimeError("D3D11 SBS surface bridge DLL is unavailable")
        self.width = int(width)
        self.height = int(height)
        self._handle = self._library.d2s_d3d11_sbs_surface_create(
            self.width, self.height, int(adapter_index)
        )
        if not self._handle:
            raise RuntimeError(_last_error(self._library))

    @property
    def device(self) -> int:
        return int(self._library.d2s_d3d11_sbs_surface_device(self._handle) or 0)

    @property
    def adapter_luid(self) -> int:
        return int(self._library.d2s_d3d11_sbs_surface_adapter_luid(self._handle))

    def upload_bgra(self, frame, *, stride: int | None = None) -> None:
        data = memoryview(frame)
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError(f"expected HWC BGRA8 frame, got shape={data.shape!r}")
        height, width, _ = data.shape
        if int(width) != self.width or int(height) != self.height:
            raise ValueError("final SBS frame dimensions changed; surface recreation is required")
        if not data.contiguous:
            data = memoryview(data.tobytes()).cast("B")
        else:
            data = data.cast("B")
        row_stride = int(stride or width * 4)
        # ctypes keeps this temporary host buffer alive for the native call;
        # the native bridge immediately copies it into its D3D11 staging
        # texture before returning.
        payload = data.tobytes()
        result = self._library.d2s_d3d11_sbs_surface_upload_bgra(
            self._handle, ctypes.c_char_p(payload), row_stride, self.width, self.height
        )
        if result != 1:
            raise RuntimeError(_last_error(self._library))

    def nv12_texture(self) -> tuple[int, int, int]:
        texture = ctypes.c_void_p()
        width = ctypes.c_int()
        height = ctypes.c_int()
        result = self._library.d2s_d3d11_sbs_surface_nv12(
            self._handle, ctypes.byref(texture), ctypes.byref(width), ctypes.byref(height)
        )
        if result != 1:
            raise RuntimeError(_last_error(self._library))
        return int(texture.value or 0), int(width.value), int(height.value)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.d2s_d3d11_sbs_surface_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "D3D11SbsSurface":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["D3D11SbsSurface", "probe_d3d11_sbs_surface"]
