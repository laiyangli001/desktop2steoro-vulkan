from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from .filament_vulkan_bridge import FilamentBridgeError


class FilamentDesktopPreview:
    """Filament desktop-window renderer used by the room layout preview."""

    def __init__(self, native_window: int, width: int, height: int, library_path=None):
        path = Path(library_path) if library_path else self._default_library_path()
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise FilamentBridgeError(f"unable to load Filament preview Bridge: {path}") from exc
        self._configure_abi()
        self._handle = ctypes.c_void_p(
            self._library.filament_preview_create(
                ctypes.c_void_p(int(native_window)), int(width), int(height)
            )
        )
        self._raise_if_error("create")

    @staticmethod
    def _default_library_path() -> Path:
        names = {
            "win32": "filament_bridge.dll",
            "darwin": "libfilament_bridge.dylib",
            "linux": "libfilament_bridge.so",
        }
        try:
            name = names[sys.platform]
        except KeyError as exc:
            raise FilamentBridgeError(f"unsupported platform: {sys.platform}") from exc
        return Path(__file__).resolve().parent / "native" / name

    def load_glb(self, data: bytes) -> None:
        payload = ctypes.create_string_buffer(bytes(data))
        self._check(
            self._library.filament_preview_load_glb(
                self._handle, payload, len(data)
            ),
            "load_glb",
        )

    def set_camera(self, eye, center, up) -> None:
        self._check(
            self._library.filament_preview_set_camera(
                self._handle, *(float(v) for v in (*eye, *center, *up))
            ),
            "set_camera",
        )

    def set_projection(self, fov_degrees, aspect, near_plane, far_plane) -> None:
        self._check(
            self._library.filament_preview_set_projection(
                self._handle, float(fov_degrees), float(aspect),
                float(near_plane), float(far_plane),
            ),
            "set_projection",
        )

    def render(self) -> None:
        self._check(self._library.filament_preview_render(self._handle), "render")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.filament_preview_destroy(self._handle)
            self._handle = None

    def _configure_abi(self) -> None:
        library = self._library
        library.filament_preview_create.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
        ]
        library.filament_preview_create.restype = ctypes.c_void_p
        library.filament_preview_destroy.argtypes = [ctypes.c_void_p]
        library.filament_preview_destroy.restype = None
        library.filament_preview_load_glb.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_preview_load_glb.restype = ctypes.c_int
        library.filament_preview_set_camera.argtypes = [
            ctypes.c_void_p,
            *([ctypes.c_float] * 9),
        ]
        library.filament_preview_set_camera.restype = ctypes.c_int
        library.filament_preview_set_projection.argtypes = [
            ctypes.c_void_p, *([ctypes.c_double] * 4)
        ]
        library.filament_preview_set_projection.restype = ctypes.c_int
        library.filament_preview_render.argtypes = [ctypes.c_void_p]
        library.filament_preview_render.restype = ctypes.c_int
        library.filament_preview_last_error.argtypes = [ctypes.c_void_p]
        library.filament_preview_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        value = self._library.filament_preview_last_error(self._handle)
        return value.decode("utf-8", errors="replace") if value else ""

    def _raise_if_error(self, operation: str) -> None:
        message = self._last_error()
        if message:
            self.close()
            raise FilamentBridgeError(f"{operation}: {message}")

    def _check(self, result: int, operation: str) -> None:
        if int(result) == 0:
            raise FilamentBridgeError(f"{operation}: {self._last_error() or 'Filament preview failed'}")
