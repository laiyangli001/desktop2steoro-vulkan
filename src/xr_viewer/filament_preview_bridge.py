from __future__ import annotations

import ctypes
import io
import json
import struct
import sys
from pathlib import Path

from .filament_vulkan_bridge import FilamentBridgeError


def prepare_glb_for_preview(data: bytes, max_texture_dimension: int = 4096) -> tuple[bytes, int]:
    """Downsample oversized embedded PNG/JPEG textures without changing the source GLB."""
    if max_texture_dimension <= 0:
        return bytes(data), 0
    if len(data) < 20 or data[:4] != b"glTF":
        raise FilamentBridgeError("preview asset is not a GLB")

    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length > len(data):
        raise FilamentBridgeError("preview GLB header is invalid")
    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        if chunk_type == 0x4E4F534A:
            json_chunk = chunk
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
        offset += chunk_length
    if json_chunk is None or bin_chunk is None:
        raise FilamentBridgeError("preview GLB is missing JSON or BIN chunk")

    document = json.loads(json_chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
    views = document.get("bufferViews", [])
    images = document.get("images", [])
    replacements = {}
    from PIL import Image

    for image in images:
        view_index = image.get("bufferView")
        mime_type = str(image.get("mimeType", ""))
        if not isinstance(view_index, int) or mime_type not in {"image/png", "image/jpeg"}:
            continue
        if view_index < 0 or view_index >= len(views):
            continue
        view = views[view_index]
        start = int(view.get("byteOffset", 0))
        length = int(view.get("byteLength", 0))
        encoded = bin_chunk[start:start + length]
        try:
            with Image.open(io.BytesIO(encoded)) as source:
                width, height = source.size
                if max(width, height) <= max_texture_dimension:
                    continue
                scale = max_texture_dimension / max(width, height)
                resized = source.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
                output = io.BytesIO()
                if mime_type == "image/png":
                    resized.save(output, format="PNG", optimize=True)
                else:
                    resized.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
                replacements[view_index] = output.getvalue()
        except Exception as exc:
            raise FilamentBridgeError(f"preview texture preprocessing failed: {exc}") from exc

    if not replacements:
        return bytes(data), 0

    ordered_views = sorted(
        ((int(view.get("byteOffset", 0)), index, view) for index, view in enumerate(views)),
        key=lambda item: item[0],
    )
    rebuilt = bytearray()
    previous_end = 0
    for start, view_index, view in ordered_views:
        length = int(view.get("byteLength", 0))
        end = start + length
        if start < previous_end or end > len(bin_chunk):
            return bytes(data), 0
        rebuilt.extend(bin_chunk[previous_end:start])
        while len(rebuilt) % 4:
            rebuilt.append(0)
        view["byteOffset"] = len(rebuilt)
        payload = replacements.get(view_index, bin_chunk[start:end])
        view["byteLength"] = len(payload)
        rebuilt.extend(payload)
        while len(rebuilt) % 4:
            rebuilt.append(0)
        previous_end = end
    rebuilt.extend(bin_chunk[previous_end:])
    while len(rebuilt) % 4:
        rebuilt.append(0)
    document.setdefault("buffers", [{}])[0]["byteLength"] = len(rebuilt)

    encoded_json = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded_json += b" " * ((4 - len(encoded_json) % 4) % 4)
    total_length = 12 + 8 + len(encoded_json) + 8 + len(rebuilt)
    output = bytearray(struct.pack("<4sII", b"glTF", 2, total_length))
    output.extend(struct.pack("<II", len(encoded_json), 0x4E4F534A))
    output.extend(encoded_json)
    output.extend(struct.pack("<II", len(rebuilt), 0x004E4942))
    output.extend(rebuilt)
    return bytes(output), len(replacements)


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

    def load_glb(self, data: bytes, max_texture_dimension: int = 4096) -> None:
        data, resized_count = prepare_glb_for_preview(data, max_texture_dimension)
        if resized_count:
            print(
                f"[FilamentPreview] resized {resized_count} oversized textures "
                f"to <= {max_texture_dimension}px",
                flush=True,
            )
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

    def apply_animations(self, time_seconds: float) -> None:
        self._check(
            self._library.filament_preview_apply_animations(
                self._handle, float(time_seconds)
            ),
            "apply_animations",
        )

    def create_star_glim_material(self) -> None:
        self._check(
            self._library.filament_preview_create_star_glim_material(self._handle),
            "create_star_glim_material",
        )

    def set_star_glim_textures(self, stars_data: bytes, mask_data: bytes) -> None:
        stars = ctypes.create_string_buffer(bytes(stars_data))
        mask = ctypes.create_string_buffer(bytes(mask_data))
        self._check(
            self._library.filament_preview_set_star_glim_textures(
                self._handle, stars, len(stars_data), mask, len(mask_data)
            ),
            "set_star_glim_textures",
        )

    def set_star_glim_parameters(self, intensity: float, speed: float, seed: float) -> None:
        self._check(
            self._library.filament_preview_set_star_glim_parameters(
                self._handle, float(intensity), float(speed), float(seed)
            ),
            "set_star_glim_parameters",
        )

    def set_star_glim_time(self, time_seconds: float) -> None:
        self._check(
            self._library.filament_preview_set_star_glim_time(
                self._handle, float(time_seconds)
            ),
            "set_star_glim_time",
        )

    def set_projection(self, fov_degrees, aspect, near_plane, far_plane) -> None:
        self._check(
            self._library.filament_preview_set_projection(
                self._handle, float(fov_degrees), float(aspect),
                float(near_plane), float(far_plane),
            ),
            "set_projection",
        )

    def set_viewport(self, width: int, height: int) -> None:
        self._check(
            self._library.filament_preview_set_viewport(
                self._handle, int(width), int(height)
            ),
            "set_viewport",
        )

    def set_exposure(self, exposure_ev: float) -> None:
        self._check(
            self._library.filament_preview_set_scene_exposure(
                self._handle, float(exposure_ev)
            ),
            "set_exposure",
        )

    def set_fill_light(self, color, intensity: float, direction) -> None:
        self._check(
            self._library.filament_preview_set_fill_light(
                self._handle,
                *(float(value) for value in (*color, intensity, *direction)),
            ),
            "set_fill_light",
        )

    def set_skybox_brightness(self, brightness: float) -> None:
        self._check(
            self._library.filament_preview_set_skybox_brightness(
                self._handle, float(brightness)
            ),
            "set_skybox_brightness",
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
        library.filament_preview_apply_animations.argtypes = [
            ctypes.c_void_p, ctypes.c_double
        ]
        library.filament_preview_apply_animations.restype = ctypes.c_int
        library.filament_preview_create_star_glim_material.argtypes = [ctypes.c_void_p]
        library.filament_preview_create_star_glim_material.restype = ctypes.c_int
        library.filament_preview_set_star_glim_textures.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_preview_set_star_glim_textures.restype = ctypes.c_int
        library.filament_preview_set_star_glim_parameters.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        library.filament_preview_set_star_glim_parameters.restype = ctypes.c_int
        library.filament_preview_set_star_glim_time.argtypes = [
            ctypes.c_void_p, ctypes.c_double
        ]
        library.filament_preview_set_star_glim_time.restype = ctypes.c_int
        library.filament_preview_set_camera.argtypes = [
            ctypes.c_void_p,
            *([ctypes.c_float] * 9),
        ]
        library.filament_preview_set_camera.restype = ctypes.c_int
        library.filament_preview_set_projection.argtypes = [
            ctypes.c_void_p, *([ctypes.c_double] * 4)
        ]
        library.filament_preview_set_projection.restype = ctypes.c_int
        library.filament_preview_set_viewport.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
        ]
        library.filament_preview_set_viewport.restype = ctypes.c_int
        library.filament_preview_set_scene_exposure.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        library.filament_preview_set_scene_exposure.restype = ctypes.c_int
        library.filament_preview_set_fill_light.argtypes = [
            ctypes.c_void_p, *([ctypes.c_float] * 7)
        ]
        library.filament_preview_set_fill_light.restype = ctypes.c_int
        library.filament_preview_set_skybox_brightness.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        library.filament_preview_set_skybox_brightness.restype = ctypes.c_int
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
