"""ctypes loader for the optional DXGI Desktop Duplication bridge."""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path

from desktop2stereo.stereo_runtime.providers.intel.native_artifacts import (
    native_dll_candidates,
)


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[4]
    return native_dll_candidates(
        "d2s_desktop_duplication.dll",
        environment_variable="D2S_DESKTOP_DUPLICATION_DLL",
        extra_directories=(
            root / "native" / "desktop_duplication",
            root / "src" / "desktop2stereo" / "capture" / "native",
            root / "src" / "desktop2stereo" / "capture" / "native" / "desktop_duplication",
        ),
    )


def _load_library():
    if platform.system() != "Windows":
        return None
    for path in _candidate_paths():
        if path.is_file():
            try:
                library = ctypes.WinDLL(str(path))
            except OSError:
                continue
            required = (
                "d2s_desktop_duplication_probe",
                "d2s_desktop_duplication_create",
                "d2s_desktop_duplication_acquire",
                "d2s_desktop_duplication_copy_frame",
                "d2s_desktop_duplication_device",
                "d2s_desktop_duplication_release_frame",
                "d2s_desktop_duplication_destroy",
            )
            if any(not hasattr(library, name) for name in required):
                continue
            return library
    return None


def _error_text(lib) -> str:
    buffer = ctypes.create_string_buffer(512)
    lib.d2s_desktop_duplication_last_error(buffer, len(buffer))
    return buffer.value.decode("utf-8", errors="replace")


def probe() -> dict:
    lib = _load_library()
    if lib is None:
        return {
            "available": False,
            "backend": "desktop_duplication",
            "reason": "native d2s_desktop_duplication.dll is not installed",
        }
    lib.d2s_desktop_duplication_probe.argtypes = []
    lib.d2s_desktop_duplication_probe.restype = ctypes.c_int
    lib.d2s_desktop_duplication_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.d2s_desktop_duplication_last_error.restype = ctypes.c_int
    ok = bool(lib.d2s_desktop_duplication_probe())
    return {
        "available": ok,
        "backend": "desktop_duplication",
        "reason": None if ok else _error_text(lib),
    }


class NativeD3D11TextureFrame:
    """Borrowed D3D11 texture whose lifetime is tied to one acquired frame."""

    resource_kind = "d3d11_texture"
    format = "BGRA8"

    def __init__(self, owner, texture, width, height, adapter_luid):
        self._owner = owner
        self.texture = texture
        self.width = int(width)
        self.height = int(height)
        self.adapter_luid = int(adapter_luid)
        self._released = False

    def release(self):
        if not self._released:
            self._owner.release_frame()
            self._released = True

    @property
    def device(self):
        return self._owner.device

    def readback_bgra(self):
        """Copy the currently borrowed texture to a tightly packed BGRA array."""
        import numpy as np

        payload = bytearray(self.width * self.height * 4)
        buffer = (ctypes.c_ubyte * len(payload)).from_buffer(payload)
        stride = ctypes.c_int()
        width = ctypes.c_int()
        height = ctypes.c_int()
        copied = self._owner._lib.d2s_desktop_duplication_copy_frame(
            self._owner._handle, buffer, len(payload),
            ctypes.byref(stride), ctypes.byref(width), ctypes.byref(height),
        )
        if copied < 0:
            raise RuntimeError(_error_text(self._owner._lib))
        if width.value != self.width or height.value != self.height or stride.value != self.width * 4:
            raise RuntimeError("Desktop Duplication readback returned unexpected frame geometry")
        return np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width, 4).copy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class NativeDesktopDuplication:
    def __init__(self, output_index: int = 0, timeout_ms: int = 16):
        self._lib = _load_library()
        if self._lib is None:
            raise RuntimeError("native d2s_desktop_duplication.dll is not installed")
        self._lib.d2s_desktop_duplication_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.d2s_desktop_duplication_create.restype = ctypes.c_void_p
        self._lib.d2s_desktop_duplication_acquire.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.d2s_desktop_duplication_acquire.restype = ctypes.c_int
        self._lib.d2s_desktop_duplication_device.argtypes = [ctypes.c_void_p]
        self._lib.d2s_desktop_duplication_device.restype = ctypes.c_void_p
        self._lib.d2s_desktop_duplication_release_frame.argtypes = [ctypes.c_void_p]
        self._lib.d2s_desktop_duplication_release_frame.restype = ctypes.c_int
        self._lib.d2s_desktop_duplication_destroy.argtypes = [ctypes.c_void_p]
        self._lib.d2s_desktop_duplication_destroy.restype = None
        self._handle = self._lib.d2s_desktop_duplication_create(output_index, timeout_ms)
        if not self._handle:
            raise RuntimeError(_error_text(self._lib))

    def acquire(self):
        # DXGI_ERROR_ACCESS_LOST is rebuilt by the native bridge. Retry once so
        # the caller does not have to tear down the whole capture session.
        for _attempt in range(2):
            texture = ctypes.c_void_p()
            width = ctypes.c_int()
            height = ctypes.c_int()
            luid = ctypes.c_uint64()
            result = self._lib.d2s_desktop_duplication_acquire(
                self._handle,
                ctypes.byref(texture),
                ctypes.byref(width),
                ctypes.byref(height),
                ctypes.byref(luid),
            )
            if result == 0:
                return None
            if result == -2 and _attempt == 0:
                continue
            if result < 0:
                raise RuntimeError(_error_text(self._lib))
            return {
                "texture": texture,
                "width": width.value,
                "height": height.value,
                "adapter_luid": luid.value,
            }
        raise RuntimeError("Desktop Duplication output recreation did not produce a frame")

    def acquire_frame(self):
        item = self.acquire()
        if item is None:
            return None
        return NativeD3D11TextureFrame(
            self,
            item["texture"],
            item["width"],
            item["height"],
            item["adapter_luid"],
        )

    @property
    def device(self):
        return self._lib.d2s_desktop_duplication_device(self._handle)

    def release_frame(self):
        if self._handle:
            self._lib.d2s_desktop_duplication_release_frame(self._handle)

    def close(self):
        if self._handle:
            self._lib.d2s_desktop_duplication_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
