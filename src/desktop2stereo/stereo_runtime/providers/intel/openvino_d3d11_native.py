"""Optional ctypes facade for the native OpenVINO/D3D11 bridge.

The bridge is intentionally opt-in: no DLL means no claim of RemoteTensor
zero-copy. The native library owns OpenVINO and D3D11 interop; Python only
passes borrowed native handles and model/input parameters.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

from .native_artifacts import native_dll_candidates


_DLL_NAME = "d2s_openvino_d3d11_bridge.dll"


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[5]
    return native_dll_candidates(
        _DLL_NAME,
        environment_variable="D2S_OPENVINO_D3D11_DLL",
        extra_directories=(
            root / "native" / "openvino_d3d11_bridge",
            root / "src" / "native" / "windows",
        ),
    )


def load_openvino_d3d11_bridge() -> ctypes.CDLL | None:
    if os.name != "nt":
        return None
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            library = ctypes.WinDLL(str(path))
        except OSError:
            continue
        try:
            capabilities_fn = library.d2s_openvino_d3d11_capabilities
        except AttributeError:
            continue
        required_exports = (
            "d2s_openvino_d3d11_create",
            "d2s_openvino_d3d11_adapter_luid",
            "d2s_openvino_d3d11_nv12_surface",
            "d2s_openvino_d3d11_set_texture",
            "d2s_openvino_d3d11_infer",
            "d2s_openvino_d3d11_output_shape",
            "d2s_openvino_d3d11_read_output",
            "d2s_openvino_d3d11_last_error",
            "d2s_openvino_d3d11_destroy",
        )
        if any(not hasattr(library, name) for name in required_exports):
            continue
        capabilities_fn.argtypes = []
        library.d2s_openvino_d3d11_capabilities.restype = ctypes.c_int
        library.d2s_openvino_d3d11_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        library.d2s_openvino_d3d11_create.restype = ctypes.c_void_p
        library.d2s_openvino_d3d11_adapter_luid.argtypes = [ctypes.c_void_p]
        library.d2s_openvino_d3d11_adapter_luid.restype = ctypes.c_ulonglong
        library.d2s_openvino_d3d11_nv12_surface.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.d2s_openvino_d3d11_nv12_surface.restype = ctypes.c_int
        library.d2s_openvino_d3d11_set_texture.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.d2s_openvino_d3d11_set_texture.restype = ctypes.c_int
        library.d2s_openvino_d3d11_infer.argtypes = [ctypes.c_void_p]
        library.d2s_openvino_d3d11_infer.restype = ctypes.c_int
        library.d2s_openvino_d3d11_output_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_int,
        ]
        library.d2s_openvino_d3d11_output_shape.restype = ctypes.c_int
        library.d2s_openvino_d3d11_read_output.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        library.d2s_openvino_d3d11_read_output.restype = ctypes.c_int
        library.d2s_openvino_d3d11_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.d2s_openvino_d3d11_last_error.restype = ctypes.c_int
        library.d2s_openvino_d3d11_destroy.argtypes = [ctypes.c_void_p]
        library.d2s_openvino_d3d11_destroy.restype = None
        return library
    return None


def probe_openvino_d3d11_bridge() -> dict[str, Any]:
    library = load_openvino_d3d11_bridge()
    if library is None:
        return {
            "available": False,
            "backend": "openvino_d3d11_remote_tensor",
            "reason": "native OpenVINO/D3D11 bridge DLL is not available",
        }
    capabilities = int(library.d2s_openvino_d3d11_capabilities())
    return {
        "available": True,
        "backend": "openvino_d3d11_remote_tensor",
        "capabilities": capabilities,
        "nv12_surface": bool(capabilities & 0x01),
        "bgra_to_nv12": bool(capabilities & 0x02),
        "reason": None
        if capabilities & 0x03 == 0x03
        else "bridge lacks the GPU BGRA8-to-NV12 conversion required by Desktop Duplication",
    }


class OpenVINOD3D11Session:
    """Owns one native model/request session; texture handles remain borrowed."""

    def __init__(self, model_path: str | os.PathLike[str], d3d11_device: int | ctypes.c_void_p) -> None:
        self._library = load_openvino_d3d11_bridge()
        if self._library is None:
            raise RuntimeError("OpenVINO/D3D11 native bridge DLL is unavailable")
        device = ctypes.c_void_p(d3d11_device) if isinstance(d3d11_device, int) else d3d11_device
        self._handle = self._library.d2s_openvino_d3d11_create(
            os.fspath(model_path).encode("utf-8"), device
        )
        if not self._handle:
            raise RuntimeError(self.last_error())
        self.adapter_luid = int(self._library.d2s_openvino_d3d11_adapter_luid(self._handle))
        if not self.adapter_luid:
            self.close()
            raise RuntimeError("native OpenVINO bridge returned no adapter LUID")

    def last_error(self) -> str:
        if not self._library or not self._handle:
            return "OpenVINO/D3D11 session is not initialized"
        buffer = ctypes.create_string_buffer(1024)
        self._library.d2s_openvino_d3d11_last_error(buffer, len(buffer))
        return buffer.value.decode("utf-8", errors="replace") or "native bridge operation failed"

    def nv12_surface(self) -> tuple[ctypes.c_void_p, int, int]:
        texture = ctypes.c_void_p()
        width = ctypes.c_int()
        height = ctypes.c_int()
        result = self._library.d2s_openvino_d3d11_nv12_surface(
            self._handle,
            ctypes.byref(texture),
            ctypes.byref(width),
            ctypes.byref(height),
        )
        if result != 1 or not texture.value:
            raise RuntimeError(self.last_error())
        return texture, int(width.value), int(height.value)

    def set_texture(self, input_name: str, texture: int | ctypes.c_void_p, width: int, height: int) -> None:
        handle = ctypes.c_void_p(texture) if isinstance(texture, int) else texture
        result = self._library.d2s_openvino_d3d11_set_texture(
            self._handle, input_name.encode("utf-8"), handle, int(width), int(height)
        )
        if result != 1:
            raise RuntimeError(self.last_error())

    def infer(self) -> None:
        if self._library.d2s_openvino_d3d11_infer(self._handle) != 0:
            raise RuntimeError(self.last_error())

    def output_shape(self) -> tuple[int, ...]:
        dims = (ctypes.c_longlong * 8)()
        rank = self._library.d2s_openvino_d3d11_output_shape(self._handle, dims, len(dims))
        if rank < 0:
            raise RuntimeError(self.last_error())
        return tuple(int(dims[index]) for index in range(rank))

    def read_output(self) -> tuple[float, ...]:
        shape = self.output_shape()
        count = 1
        for dimension in shape:
            count *= int(dimension)
        values = (ctypes.c_float * max(1, count))()
        actual = self._library.d2s_openvino_d3d11_read_output(
            self._handle, values, count
        )
        if actual < 0:
            raise RuntimeError(self.last_error())
        return tuple(float(values[index]) for index in range(actual))

    def infer_output(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        self.infer()
        return self.output_shape(), self.read_output()

    def close(self) -> None:
        if self._library is not None and self._handle:
            self._library.d2s_openvino_d3d11_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "OpenVINOD3D11Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = [
    "OpenVINOD3D11Session",
    "load_openvino_d3d11_bridge",
    "probe_openvino_d3d11_bridge",
]
