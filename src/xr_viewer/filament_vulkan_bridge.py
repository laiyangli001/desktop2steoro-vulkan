from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any, Iterable


class FilamentBridgeError(RuntimeError):
    pass


class _VulkanCreateInfo(ctypes.Structure):
    _fields_ = [
        ("instance", ctypes.c_void_p),
        ("physical_device", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
        ("graphics_queue_family_index", ctypes.c_uint32),
        ("graphics_queue_index", ctypes.c_uint32),
    ]


def default_bridge_path() -> Path:
    names = {
        "win32": "filament_bridge.dll",
        "darwin": "libfilament_bridge.dylib",
        "linux": "libfilament_bridge.so",
    }
    try:
        name = names[sys.platform]
    except KeyError as exc:
        raise FilamentBridgeError(
            f"unsupported platform for Filament Bridge: {sys.platform}"
        ) from exc
    return Path(__file__).resolve().parent / "native" / name


class FilamentVulkanBridge:
    """ctypes wrapper for the Python-owned OpenXR Vulkan session handles."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        path = Path(library_path) if library_path else default_bridge_path()
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise FilamentBridgeError(f"unable to load Filament Bridge: {path}") from exc
        self._configure_abi()
        self._handle: ctypes.c_void_p | None = None

    @property
    def handle(self) -> int:
        return int(self._handle.value or 0) if self._handle is not None else 0

    @property
    def loaded(self) -> bool:
        return self._handle is not None and bool(self._handle.value)

    def create(
        self,
        *,
        instance: Any,
        physical_device: Any,
        device: Any,
        queue_family_index: int,
        queue_index: int = 0,
    ) -> None:
        if self.loaded:
            return
        info = _VulkanCreateInfo(
            instance=_as_pointer_value(instance),
            physical_device=_as_pointer_value(physical_device),
            device=_as_pointer_value(device),
            graphics_queue_family_index=int(queue_family_index),
            graphics_queue_index=int(queue_index),
        )
        handle = self._library.filament_bridge_create_vulkan(ctypes.byref(info))
        if not handle:
            raise FilamentBridgeError("Filament Bridge returned a null handle")
        self._handle = ctypes.c_void_p(handle)
        self._raise_if_error()

    def create_swapchain(
        self,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
    ) -> None:
        self._ensure_loaded()
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if not values:
            raise ValueError("Filament swapchain requires at least one VkImage")
        array_type = ctypes.c_void_p * len(values)
        result = self._library.filament_bridge_create_swapchain(
            self._handle,
            array_type(*values),
            len(values),
            int(format),
            int(width),
            int(height),
        )
        self._check_result(result, "create_swapchain")

    def set_acquired_image(self, image_index: int) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_acquired_image(
                self._handle, int(image_index)
            ),
            "set_acquired_image",
        )

    def set_camera_look_at(
        self,
        eye: tuple[float, float, float],
        center: tuple[float, float, float],
        up: tuple[float, float, float],
    ) -> None:
        self._ensure_loaded()
        values = tuple(float(value) for value in (*eye, *center, *up))
        self._check_result(
            self._library.filament_bridge_set_camera_look_at(
                self._handle, *values
            ),
            "set_camera_look_at",
        )

    def set_camera_projection(
        self,
        vertical_fov_degrees: float,
        aspect: float,
        *,
        near_plane: float = 0.05,
        far_plane: float = 1000.0,
    ) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_camera_projection(
                self._handle,
                float(vertical_fov_degrees),
                float(aspect),
                float(near_plane),
                float(far_plane),
            ),
            "set_camera_projection",
        )

    def begin_frame(self) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_begin_frame(self._handle), "begin_frame"
        )

    def end_frame(self) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_end_frame(self._handle), "end_frame"
        )

    def load_glb(self, data: bytes | bytearray | memoryview) -> None:
        self._ensure_loaded()
        payload = bytes(data)
        if not payload:
            raise ValueError("GLB payload must not be empty")
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_load_glb(
                self._handle, buffer, len(payload)
            ),
            "load_glb",
        )

    def apply_animations(self, time_seconds: float) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_apply_animations(
                self._handle, float(time_seconds)
            ),
            "apply_animations",
        )

    def close(self) -> None:
        if self._handle is not None:
            self._library.filament_bridge_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "FilamentVulkanBridge":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _configure_abi(self) -> None:
        library = self._library
        library.filament_bridge_create_vulkan.argtypes = [
            ctypes.POINTER(_VulkanCreateInfo)
        ]
        library.filament_bridge_create_vulkan.restype = ctypes.c_void_p
        library.filament_bridge_destroy.argtypes = [ctypes.c_void_p]
        library.filament_bridge_destroy.restype = None
        library.filament_bridge_create_swapchain.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.filament_bridge_create_swapchain.restype = ctypes.c_int
        library.filament_bridge_set_acquired_image.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_set_acquired_image.restype = ctypes.c_int
        library.filament_bridge_set_camera_look_at.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        library.filament_bridge_set_camera_look_at.restype = ctypes.c_int
        library.filament_bridge_set_camera_projection.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
        ]
        library.filament_bridge_set_camera_projection.restype = ctypes.c_int
        for name in ("filament_bridge_begin_frame", "filament_bridge_end_frame"):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        library.filament_bridge_load_glb.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_load_glb.restype = ctypes.c_int
        library.filament_bridge_apply_animations.argtypes = [
            ctypes.c_void_p, ctypes.c_double
        ]
        library.filament_bridge_apply_animations.restype = ctypes.c_int
        library.filament_bridge_last_error.argtypes = [ctypes.c_void_p]
        library.filament_bridge_last_error.restype = ctypes.c_char_p

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            raise FilamentBridgeError("Filament Bridge is not initialized")

    def _raise_if_error(self) -> None:
        message = self._last_error()
        if message:
            self.close()
            raise FilamentBridgeError(message)

    def _check_result(self, result: int, operation: str) -> None:
        if int(result) == 0:
            message = self._last_error() or "Filament Bridge operation failed"
            raise FilamentBridgeError(f"{operation}: {message}")

    def _last_error(self) -> str:
        value = self._library.filament_bridge_last_error(self._handle)
        return value.decode("utf-8", errors="replace") if value else ""


def _as_pointer_value(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    try:
        import vulkan as vk

        return int(vk.ffi.cast("uintptr_t", value))
    except (ImportError, TypeError, ValueError):
        return int(ctypes.cast(value, ctypes.c_void_p).value or 0)
