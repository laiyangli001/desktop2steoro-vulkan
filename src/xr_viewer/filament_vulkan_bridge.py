from __future__ import annotations

import ctypes
import sys
import threading
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
    platforms = {
        "win32": ("windows", "filament_bridge.dll"),
        "darwin": ("macos", "libfilament_bridge.dylib"),
        "linux": ("linux", "libfilament_bridge.so"),
    }
    try:
        platform_dir, name = platforms[sys.platform]
    except KeyError as exc:
        raise FilamentBridgeError(
            f"unsupported platform for Filament Bridge: {sys.platform}"
        ) from exc
    return Path(__file__).resolve().parent / "native" / platform_dir / name


class FilamentVulkanBridge:
    """ctypes wrapper for the Python-owned OpenXR Vulkan session handles."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        path = Path(library_path) if library_path else default_bridge_path()
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise FilamentBridgeError(f"unable to load Filament Bridge: {path}") from exc
        self._controller_abi_available = False
        self._controller_visibility_abi_available = False
        self._laser_abi_available = False
        self._controller_guide_abi_available = False
        self._text_overlay_abi_available = False
        self._screen_image_abi_available = False
        self._screen_source_version_abi_available = False
        self._fixed_screen_image_abi_available = False
        self._vulkan_external_image_abi_available = False
        self._screen_curved_abi_available = False
        self._screen_light_abi_available = False
        self._glow_abi_available = False
        self._glow_vulkan_image_abi_available = False
        self._passthrough_backdrop_abi_available = False
        self._ambient_light_abi_available = False
        self._controller_ambient_light_abi_available = False
        self._screen_ready_semaphore_abi_available = False
        self._finished_drawing_semaphore_abi_available = False
        self._screen_sampling_abi_available = False
        self._screen_upscale_abi_available = False
        self._screen_sampling_mode_abi_available = False
        self._screen_sampling_stats_abi_available = False
        self._async_submit_abi_available = False
        self._stereo_batch_submit_abi_available = False
        self._screen_eye_renderables_abi_available = False
        self._multiview_abi_version = 0
        self._configure_abi()
        self._handle: ctypes.c_void_p | None = None
        self._owner_thread_id: int | None = None

    @property
    def handle(self) -> int:
        return int(self._handle.value or 0) if self._handle is not None else 0

    @property
    def controller_abi_available(self) -> bool:
        return self._controller_abi_available

    @property
    def laser_abi_available(self) -> bool:
        return self._laser_abi_available

    @property
    def controller_guide_abi_available(self) -> bool:
        return self._controller_guide_abi_available

    @property
    def text_overlay_abi_available(self) -> bool:
        return self._text_overlay_abi_available

    @property
    def controller_visibility_abi_available(self) -> bool:
        return self._controller_visibility_abi_available

    @property
    def screen_image_abi_available(self) -> bool:
        return self._screen_image_abi_available

    @property
    def vulkan_external_image_abi_available(self) -> bool:
        return self._vulkan_external_image_abi_available

    @property
    def screen_curved_abi_available(self) -> bool:
        return self._screen_curved_abi_available

    @property
    def passthrough_backdrop_abi_available(self) -> bool:
        return self._passthrough_backdrop_abi_available

    @property
    def screen_ready_semaphore_abi_available(self) -> bool:
        return self._screen_ready_semaphore_abi_available

    @property
    def async_submit_abi_available(self) -> bool:
        return self._async_submit_abi_available

    @property
    def stereo_batch_submit_abi_available(self) -> bool:
        return self._stereo_batch_submit_abi_available

    @property
    def screen_eye_renderables_abi_available(self) -> bool:
        return self._screen_eye_renderables_abi_available

    @property
    def multiview_abi_available(self) -> bool:
        return self._multiview_abi_version >= 2

    @property
    def multiview_supported(self) -> bool:
        if not self.multiview_abi_available or not self.loaded:
            return False
        return bool(
            self._library.filament_bridge_multiview_supported(self._handle)
        )

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
            self._ensure_owner_thread()
            return
        self._owner_thread_id = threading.get_ident()
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

    def create_eye_swapchain(
        self,
        eye_index: int,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
    ) -> None:
        self._ensure_loaded()
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if int(eye_index) not in (0, 1) or not values:
            raise ValueError("eye_index must be 0 or 1 and swapchain must not be empty")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            self._library.filament_bridge_create_eye_swapchain(
                self._handle,
                int(eye_index),
                array_type(*values),
                len(values),
                int(format),
                int(width),
                int(height),
            ),
            "create_eye_swapchain",
        )

    def create_stereo_swapchain(
        self,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
    ) -> None:
        self._ensure_loaded()
        if not self.multiview_supported:
            raise FilamentBridgeError("Filament Vulkan multiview is unavailable")
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if not values:
            raise ValueError("Filament stereo swapchain requires at least one VkImage")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            self._library.filament_bridge_create_stereo_swapchain(
                self._handle,
                array_type(*values),
                len(values),
                int(format),
                int(width),
                int(height),
            ),
            "create_stereo_swapchain",
        )

    def set_active_eye(self, eye_index: int) -> None:
        self._ensure_loaded()
        if int(eye_index) not in (0, 1):
            raise ValueError("eye_index must be 0 or 1")
        self._check_result(
            self._library.filament_bridge_set_active_eye(self._handle, int(eye_index)),
            "set_active_eye",
        )

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

    def set_camera_projection_frustum(
        self,
        left: float,
        right: float,
        bottom: float,
        top: float,
        *,
        near_plane: float = 0.05,
        far_plane: float = 1000.0,
    ) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_camera_projection_frustum(
                self._handle,
                float(left), float(right), float(bottom), float(top),
                float(near_plane), float(far_plane),
            ),
            "set_camera_projection_frustum",
        )

    def set_stereo_camera(
        self,
        eye_model_matrices: Iterable[float],
        eye_frustums: Iterable[float],
        *,
        near_plane: float = 0.05,
        far_plane: float = 1000.0,
    ) -> None:
        self._ensure_loaded()
        if not self.multiview_abi_available:
            raise FilamentBridgeError("Filament Vulkan multiview ABI is unavailable")
        matrices = tuple(float(value) for value in eye_model_matrices)
        frustums = tuple(float(value) for value in eye_frustums)
        if len(matrices) != 32 or len(frustums) != 8:
            raise ValueError("stereo camera requires 32 matrix and 8 frustum values")
        self._check_result(
            self._library.filament_bridge_set_stereo_camera(
                self._handle,
                (ctypes.c_float * 32)(*matrices),
                (ctypes.c_double * 8)(*frustums),
                float(near_plane),
                float(far_plane),
            ),
            "set_stereo_camera",
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

    def end_frame_deferred(self) -> None:
        self._ensure_loaded()
        if not self._stereo_batch_submit_abi_available:
            raise FilamentBridgeError("stereo batch submit ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_end_frame_deferred(self._handle),
            "end_frame_deferred",
        )

    def finish_frame_batch(self) -> None:
        self._ensure_loaded()
        if not self._stereo_batch_submit_abi_available:
            raise FilamentBridgeError("stereo batch submit ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_finish_frame_batch(self._handle),
            "finish_frame_batch",
        )

    def wait_for_idle(self) -> None:
        self._ensure_loaded()
        if not self._async_submit_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_wait_for_idle(self._handle),
            "wait_for_idle",
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

    def load_controller(self, hand: int, data: bytes | bytearray | memoryview) -> None:
        self._ensure_loaded()
        self._ensure_controller_abi()
        payload = bytes(data)
        if int(hand) not in (0, 1) or not payload:
            raise ValueError("controller hand must be 0 or 1 and payload must not be empty")
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_load_controller(
                self._handle, int(hand), buffer, len(payload)
            ),
            "load_controller",
        )

    def set_controller_pose(self, hand: int, matrix) -> None:
        self._ensure_loaded()
        self._ensure_controller_abi()
        values = [float(value) for value in matrix.reshape(-1, order="F")]
        if int(hand) not in (0, 1) or len(values) != 16:
            raise ValueError("controller pose must be a 4x4 matrix")
        array_type = ctypes.c_float * 16
        self._check_result(
            self._library.filament_bridge_set_controller_pose(
                self._handle, int(hand), array_type(*values)
            ),
            "set_controller_pose",
        )

    def set_controller_inputs(
        self,
        hand: int,
        *,
        trigger: float,
        grip: float,
        joystick_x: float,
        joystick_y: float,
        button_mask: int,
    ) -> None:
        self._ensure_loaded()
        self._ensure_controller_abi()
        self._check_result(
            self._library.filament_bridge_set_controller_inputs(
                self._handle,
                int(hand),
                float(trigger),
                float(grip),
                float(joystick_x),
                float(joystick_y),
                int(button_mask),
            ),
            "set_controller_inputs",
        )

    def set_controller_visible(self, hand: int, visible: bool) -> None:
        self._ensure_loaded()
        if not self._controller_visibility_abi_available:
            return
        self._ensure_controller_abi()
        self._check_result(
            self._library.filament_bridge_set_controller_visible(
                self._handle, int(hand), int(bool(visible))
            ),
            "set_controller_visible",
        )

    def set_controller_laser(self, hand: int, matrix, *, visible: bool) -> None:
        self._ensure_loaded()
        if not self._laser_abi_available:
            return
        self._ensure_controller_abi()
        values = [float(value) for value in matrix.reshape(-1, order="F")]
        if int(hand) not in (0, 1) or len(values) != 16:
            raise ValueError("controller laser matrix must be 4x4")
        array_type = ctypes.c_float * 16
        self._check_result(
            self._library.filament_bridge_set_controller_laser(
                self._handle, int(hand), array_type(*values), int(bool(visible))
            ),
            "set_controller_laser",
        )

    def set_controller_guide_texture(self, rgba) -> None:
        self._ensure_loaded()
        if not self._controller_guide_abi_available:
            return
        if getattr(rgba, "ndim", 0) != 3 or rgba.shape[2] != 4:
            raise ValueError("controller guide texture must be HxWx4 RGBA")
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        payload = bytes(memoryview(rgba).cast("B"))
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_set_controller_guide_texture(
                self._handle, buffer, width, height
            ),
            "set_controller_guide_texture",
        )

    def set_controller_guide(self, matrix, *, visible: bool) -> None:
        self._ensure_loaded()
        if not self._controller_guide_abi_available:
            return
        values = [float(value) for value in matrix.reshape(-1, order="F")]
        if len(values) != 16:
            raise ValueError("controller guide pose must be a 4x4 matrix")
        array_type = ctypes.c_float * 16
        self._check_result(
            self._library.filament_bridge_set_controller_guide(
                self._handle, array_type(*values), int(bool(visible))
            ),
            "set_controller_guide",
        )

    def set_text_overlay_page_texture(self, page: int, rgba) -> None:
        """Upload one MSDF atlas page; the native Bridge owns the GPU copy."""
        self._ensure_loaded()
        if not self._text_overlay_abi_available:
            return
        if int(page) < 0 or int(page) >= 4:
            raise ValueError("MSDF atlas page must be in the range 0..3")
        if getattr(rgba, "ndim", 0) != 3 or rgba.shape[2] != 4:
            raise ValueError("MSDF atlas page must be HxWx4 RGBA")
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        payload = _array_payload(rgba)
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_set_text_overlay_page_texture(
                self._handle, int(page), buffer, width, height
            ),
            "set_text_overlay_page_texture",
        )

    def set_text_overlay_page(
        self, page: int, vertices, indices, *, visible: bool
    ) -> None:
        """Submit packed world-space MSDF vertices on the Presenter thread."""
        self._ensure_loaded()
        if not self._text_overlay_abi_available:
            return
        if getattr(vertices, "ndim", 0) != 2 or vertices.shape[1] != 9:
            raise ValueError("MSDF vertices must have shape (N, 9)")
        if getattr(indices, "ndim", 0) != 1:
            raise ValueError("MSDF indices must be a one-dimensional array")
        if getattr(vertices, "itemsize", 0) != ctypes.sizeof(ctypes.c_float):
            raise ValueError("MSDF vertices must use float32 storage")
        if getattr(indices, "itemsize", 0) != ctypes.sizeof(ctypes.c_uint16):
            raise ValueError("MSDF indices must use uint16 storage")
        if hasattr(vertices, "flags") and not vertices.flags.c_contiguous:
            raise ValueError("MSDF vertices must be contiguous")
        if hasattr(indices, "flags") and not indices.flags.c_contiguous:
            raise ValueError("MSDF indices must be contiguous")
        vertex_payload = _array_payload(vertices)
        index_payload = _array_payload(indices)
        vertex_buffer = ctypes.create_string_buffer(vertex_payload)
        index_buffer = ctypes.create_string_buffer(index_payload)
        self._check_result(
            self._library.filament_bridge_set_text_overlay_page_vertices(
                self._handle,
                int(page),
                vertex_buffer,
                int(vertices.shape[0]),
                index_buffer,
                int(indices.shape[0]),
                int(bool(visible)),
            ),
            "set_text_overlay_page",
        )

    def set_scene_exposure(self, exposure_ev: float) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_scene_exposure(
                self._handle, float(exposure_ev)
            ),
            "set_scene_exposure",
        )

    def set_skybox_brightness(self, brightness: float) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_skybox_brightness(
                self._handle, float(brightness)
            ),
            "set_skybox_brightness",
        )

    def set_passthrough_backdrop(self, enabled: bool) -> None:
        self._ensure_loaded()
        if not self._passthrough_backdrop_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge passthrough-backdrop ABI is unavailable; rebuild the CI artifact"
            )
        self._check_result(
            self._library.filament_bridge_set_passthrough_backdrop(
                self._handle, int(bool(enabled))
            ),
            "set_passthrough_backdrop",
        )

    def set_fill_light(self, color, intensity: float, direction) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_fill_light(
                self._handle,
                *(float(value) for value in (*color, intensity, *direction)),
            ),
            "set_fill_light",
        )

    def set_ambient_light(self, color) -> None:
        self._ensure_loaded()
        if not self._ambient_light_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_ambient_light(
                self._handle, *(float(value) for value in color)
            ),
            "set_ambient_light",
        )

    def set_controller_ambient_light(self, color, enabled: bool) -> None:
        self._ensure_loaded()
        if not self._controller_ambient_light_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_controller_ambient_light(
                self._handle,
                *(float(value) for value in color),
                int(bool(enabled)),
            ),
            "set_controller_ambient_light",
        )

    def create_screen(self) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_create_screen(self._handle),
            "create_screen",
        )

    def set_screen(self, position, width: float, height: float, rotation_deg) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_screen(
                self._handle,
                *(float(value) for value in (*position, width, height, *rotation_deg)),
            ),
            "set_screen",
        )

    def set_screen_curved(self, curved: bool) -> None:
        self._ensure_loaded()
        if not self._screen_curved_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge curved-screen ABI is unavailable; rebuild the CI artifact"
            )
        self._check_result(
            self._library.filament_bridge_set_screen_curved(
                self._handle, int(bool(curved))
            ),
            "set_screen_curved",
        )

    def set_screen_light(self, color, intensity: float) -> None:
        self._ensure_loaded()
        if not self._screen_light_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_screen_light(
                self._handle,
                *(float(value) for value in color),
                float(intensity),
            ),
            "set_screen_light",
        )

    @property
    def glow_abi_available(self) -> bool:
        return self._glow_abi_available

    @property
    def glow_vulkan_image_abi_available(self) -> bool:
        return self._glow_vulkan_image_abi_available

    def set_glow_source(self, rgba: bytes, *, width: int, height: int) -> None:
        """Upload the legacy glow source through an ordinary CPU sRGB texture."""
        self._ensure_loaded()
        if not self._glow_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge glow ABI is unavailable; rebuild the CI artifact"
            )
        expected = int(width) * int(height) * 4
        payload = bytes(rgba)
        if int(width) <= 0 or int(height) <= 0 or len(payload) != expected:
            raise ValueError("glow source must contain width*height*4 RGBA bytes")
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_set_glow_source(
                self._handle, buffer, int(width), int(height)
            ),
            "set_glow_source",
        )

    def set_glow_image(self, resource: Any) -> None:
        """Bind a completed presenter-owned Vulkan image as the Glow source."""
        self._ensure_loaded()
        if not self._glow_vulkan_image_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge Glow external-image ABI is unavailable; rebuild the CI artifact"
            )
        image = getattr(resource, "image", None)
        width = int(getattr(resource, "width", 0) or 0)
        height = int(getattr(resource, "height", 0) or 0)
        format_value = int(getattr(resource, "format", 0) or 0)
        if image is None or width <= 0 or height <= 0 or format_value <= 0:
            raise ValueError("Glow Vulkan image resource is incomplete")
        self._check_result(
            self._library.filament_bridge_set_glow_image(
                self._handle,
                ctypes.c_void_p(_as_pointer_value(image)),
                width,
                height,
                format_value,
            ),
            "set_glow_image",
        )

    def set_glow_state(
        self,
        mode: str,
        head_position,
        *,
        glow_intensity: float,
        glow_width: float,
        glow_intensity_multiplier: float,
        frosted_intensity: float,
        frosted_alpha: float,
        frosted_threshold: float,
        frosted_lod: float,
        frosted_blend: float,
        frosted_thickness: float,
        frosted_diffuse: float,
        frosted_inset: float,
        veil_intensity: float,
        veil_alpha: float,
        glow_shell_intensity_multiplier: float,
        glow_shell_radius: float,
        glow_shell_height: float,
    ) -> None:
        """Apply the v2.5 OpenXR Glow state to its dedicated Vulkan source."""
        self._ensure_loaded()
        if not self._glow_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge glow ABI is unavailable; rebuild the CI artifact"
            )
        normalized = str(mode or "off").strip().lower()
        mode_value = {
            "off": 0,
            "glow": 1,
            "screen": 1,
            "glow2": 2,
            "veil": 3,
            "frosted": 4,
            "frost": 4,
            "surround": 5,
        }.get(normalized)
        if mode_value is None:
            raise ValueError(f"unsupported glow mode: {mode}")
        values = (
            *(float(value) for value in head_position),
            float(glow_intensity),
            float(glow_width),
            float(glow_intensity_multiplier),
            float(frosted_intensity),
            float(frosted_alpha),
            float(frosted_threshold),
            float(frosted_lod),
            float(frosted_blend),
            float(frosted_thickness),
            float(frosted_diffuse),
            float(frosted_inset),
            float(veil_intensity),
            float(veil_alpha),
            float(glow_shell_intensity_multiplier),
            float(glow_shell_radius),
            float(glow_shell_height),
        )
        if len(values) != 19:
            raise ValueError("head_position must contain exactly three values")
        self._check_result(
            self._library.filament_bridge_set_glow_state(
                self._handle, ctypes.c_uint32(mode_value), *values
            ),
            "set_glow_state",
        )

    def set_screen_sampling(self, filter_scale: float) -> None:
        self._ensure_loaded()
        if not self._screen_sampling_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_screen_sampling(
                self._handle, float(filter_scale)
            ),
            "set_screen_sampling",
        )

    def set_screen_upscale(self, upscale_scale: float) -> None:
        self._ensure_loaded()
        set_upscale = getattr(
            self._library, "filament_bridge_set_screen_upscale", None
        )
        if set_upscale is None:
            return
        self._check_result(
            set_upscale(self._handle, float(upscale_scale)),
            "set_screen_upscale",
        )

    @property
    def screen_upscale_abi_available(self) -> bool:
        return self._screen_upscale_abi_available

    def set_screen_sampling_mode(self, mode: str) -> None:
        """Select the legacy external sampler or the internal MIP path."""
        self._ensure_loaded()
        normalized = str(mode).strip().lower()
        if normalized not in {"legacy", "mip"}:
            raise ValueError("screen sampling mode must be 'legacy' or 'mip'")
        if not self._screen_sampling_mode_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge screen sampling mode ABI is unavailable; rebuild the CI artifact"
            )
        self._check_result(
            self._library.filament_bridge_set_screen_sampling_mode(
                self._handle, int(normalized == "mip")
            ),
            "set_screen_sampling_mode",
        )

    @property
    def screen_sampling_abi_available(self) -> bool:
        return self._screen_sampling_abi_available

    @property
    def screen_sampling_mode_abi_available(self) -> bool:
        return self._screen_sampling_mode_abi_available

    @property
    def screen_sampling_stats_abi_available(self) -> bool:
        return self._screen_sampling_stats_abi_available

    def screen_sampling_stats(self, eye_index: int) -> dict[str, int]:
        """Return dynamic virtual-screen sampling counters for one eye."""
        self._ensure_loaded()
        if not self._screen_sampling_stats_abi_available:
            return {}
        eye = int(eye_index)
        if eye not in (0, 1):
            raise ValueError("eye_index must be 0 or 1")
        source_binds = ctypes.c_uint64()
        mip_generations = ctypes.c_uint64()
        self._check_result(
            self._library.filament_bridge_get_screen_sampling_stats(
                self._handle,
                ctypes.c_uint32(eye),
                ctypes.byref(source_binds),
                ctypes.byref(mip_generations),
            ),
            "get_screen_sampling_stats",
        )
        return {
            "source_binds": int(source_binds.value),
            "mip_generations": int(mip_generations.value),
        }

    def set_screen_image(
        self, image: Any, *, width: int, height: int, format: int
    ) -> None:
        """Bind a borrowed Vulkan image as the virtual screen texture."""
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_screen_image(
                self._handle,
                ctypes.c_void_p(_as_pointer_value(image)),
                int(width),
                int(height),
                int(format),
            ),
            "set_screen_image",
        )

    def set_screen_source_version(self, version: int) -> None:
        """Mark the active eye's source version to reuse unchanged MIPs."""
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_screen_source_version(
                self._handle,
                int(version),
            ),
            "set_screen_source_version",
        )

    def set_fixed_screen_image(self, rgba) -> None:
        """Upload one deterministic RGBA image for screen A/B regression."""
        self._ensure_loaded()
        if not self._fixed_screen_image_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge fixed screen image ABI is unavailable; rebuild the CI artifact"
            )
        if getattr(rgba, "ndim", 0) != 3 or rgba.shape[2] != 4:
            raise ValueError("fixed screen image must be HxWx4 RGBA")
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        payload = _array_payload(rgba)
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_set_fixed_screen_image(
                self._handle, buffer, width, height
            ),
            "set_fixed_screen_image",
        )

    def set_screen_ready_semaphore(self, semaphore: Any) -> None:
        self._ensure_loaded()
        if not self._screen_ready_semaphore_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_screen_ready_semaphore(
                self._handle, ctypes.c_void_p(_as_pointer_value(semaphore))
            ),
            "set_screen_ready_semaphore",
        )

    @property
    def finished_drawing_semaphore_abi_available(self) -> bool:
        return self._finished_drawing_semaphore_abi_available

    def get_finished_drawing_semaphore(self) -> int | None:
        self._ensure_loaded()
        if not self._finished_drawing_semaphore_abi_available:
            return None
        output = ctypes.c_void_p()
        result = self._library.filament_bridge_get_finished_drawing_semaphore(
            self._handle, ctypes.byref(output)
        )
        if int(result) == 0 or not output.value:
            return None
        return int(output.value)

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
            self._ensure_owner_thread()
            self._library.filament_bridge_destroy(self._handle)
            self._handle = None
        self._owner_thread_id = None

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
        library.filament_bridge_create_eye_swapchain.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.filament_bridge_create_eye_swapchain.restype = ctypes.c_int
        multiview_abi = getattr(
            library, "filament_bridge_multiview_abi_available", None
        )
        multiview_supported = getattr(
            library, "filament_bridge_multiview_supported", None
        )
        create_stereo_swapchain = getattr(
            library, "filament_bridge_create_stereo_swapchain", None
        )
        set_stereo_camera = getattr(
            library, "filament_bridge_set_stereo_camera", None
        )
        if all(
            function is not None
            for function in (
                multiview_abi,
                multiview_supported,
                create_stereo_swapchain,
                set_stereo_camera,
            )
        ):
            multiview_abi.argtypes = []
            multiview_abi.restype = ctypes.c_int
            multiview_supported.argtypes = [ctypes.c_void_p]
            multiview_supported.restype = ctypes.c_int
            create_stereo_swapchain.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint32,
                ctypes.c_int32,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            create_stereo_swapchain.restype = ctypes.c_int
            set_stereo_camera.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
            ]
            set_stereo_camera.restype = ctypes.c_int
            self._multiview_abi_version = int(multiview_abi())
        library.filament_bridge_set_active_eye.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_set_active_eye.restype = ctypes.c_int
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
        library.filament_bridge_set_camera_projection_frustum.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
        ]
        library.filament_bridge_set_camera_projection_frustum.restype = ctypes.c_int
        for name in ("filament_bridge_begin_frame", "filament_bridge_end_frame"):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        deferred_end_frame = getattr(
            library, "filament_bridge_end_frame_deferred", None
        )
        finish_frame_batch = getattr(
            library, "filament_bridge_finish_frame_batch", None
        )
        if deferred_end_frame is not None and finish_frame_batch is not None:
            deferred_end_frame.argtypes = [ctypes.c_void_p]
            deferred_end_frame.restype = ctypes.c_int
            finish_frame_batch.argtypes = [ctypes.c_void_p]
            finish_frame_batch.restype = ctypes.c_int
            self._stereo_batch_submit_abi_available = True
        screen_eye_renderables = getattr(
            library, "filament_bridge_screen_eye_renderables_abi_available", None
        )
        if screen_eye_renderables is not None:
            screen_eye_renderables.argtypes = []
            screen_eye_renderables.restype = ctypes.c_int
            self._screen_eye_renderables_abi_available = bool(
                screen_eye_renderables()
            )
        wait_for_idle = getattr(library, "filament_bridge_wait_for_idle", None)
        if wait_for_idle is not None:
            wait_for_idle.argtypes = [ctypes.c_void_p]
            wait_for_idle.restype = ctypes.c_int
            self._async_submit_abi_available = True
        library.filament_bridge_load_glb.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_load_glb.restype = ctypes.c_int
        controller_functions = (
            "filament_bridge_load_controller",
            "filament_bridge_set_controller_pose",
            "filament_bridge_set_controller_inputs",
        )
        if all(hasattr(library, name) for name in controller_functions):
            library.filament_bridge_load_controller.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
            ]
            library.filament_bridge_load_controller.restype = ctypes.c_int
            library.filament_bridge_set_controller_pose.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)
            ]
            library.filament_bridge_set_controller_pose.restype = ctypes.c_int
            library.filament_bridge_set_controller_inputs.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint32,
            ]
            library.filament_bridge_set_controller_inputs.restype = ctypes.c_int
            self._controller_abi_available = True
        set_controller_visible = getattr(
            library, "filament_bridge_set_controller_visible", None
        )
        if set_controller_visible is not None:
            set_controller_visible.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
            ]
            set_controller_visible.restype = ctypes.c_int
            self._controller_visibility_abi_available = True
        set_controller_laser = getattr(
            library, "filament_bridge_set_controller_laser", None
        )
        if set_controller_laser is not None:
            set_controller_laser.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int
            ]
            set_controller_laser.restype = ctypes.c_int
            self._laser_abi_available = True
        guide_texture = getattr(
            library, "filament_bridge_set_controller_guide_texture", None
        )
        guide_pose = getattr(library, "filament_bridge_set_controller_guide", None)
        if guide_texture is not None and guide_pose is not None:
            guide_texture.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32,
            ]
            guide_texture.restype = ctypes.c_int
            guide_pose.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
            ]
            guide_pose.restype = ctypes.c_int
            self._controller_guide_abi_available = True
        text_texture = getattr(
            library, "filament_bridge_set_text_overlay_page_texture", None
        )
        text_vertices = getattr(
            library, "filament_bridge_set_text_overlay_page_vertices", None
        )
        if text_texture is not None and text_vertices is not None:
            text_texture.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32,
            ]
            text_texture.restype = ctypes.c_int
            text_vertices.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_int,
            ]
            text_vertices.restype = ctypes.c_int
            self._text_overlay_abi_available = True
        library.filament_bridge_set_scene_exposure.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        library.filament_bridge_set_scene_exposure.restype = ctypes.c_int
        library.filament_bridge_set_skybox_brightness.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        library.filament_bridge_set_skybox_brightness.restype = ctypes.c_int
        set_passthrough_backdrop = getattr(
            library, "filament_bridge_set_passthrough_backdrop", None
        )
        if set_passthrough_backdrop is not None:
            set_passthrough_backdrop.argtypes = [ctypes.c_void_p, ctypes.c_int]
            set_passthrough_backdrop.restype = ctypes.c_int
            self._passthrough_backdrop_abi_available = True
        set_ambient_light = getattr(
            library, "filament_bridge_set_ambient_light", None
        )
        if set_ambient_light is not None:
            set_ambient_light.argtypes = [
                ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float
            ]
            set_ambient_light.restype = ctypes.c_int
            self._ambient_light_abi_available = True
        set_controller_ambient_light = getattr(
            library, "filament_bridge_set_controller_ambient_light", None
        )
        if set_controller_ambient_light is not None:
            set_controller_ambient_light.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int,
            ]
            set_controller_ambient_light.restype = ctypes.c_int
            self._controller_ambient_light_abi_available = True
        library.filament_bridge_set_fill_light.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        library.filament_bridge_set_fill_light.restype = ctypes.c_int
        library.filament_bridge_create_screen.argtypes = [ctypes.c_void_p]
        library.filament_bridge_create_screen.restype = ctypes.c_int
        library.filament_bridge_set_screen.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        library.filament_bridge_set_screen.restype = ctypes.c_int
        set_screen_curved = getattr(
            library, "filament_bridge_set_screen_curved", None
        )
        if set_screen_curved is not None:
            set_screen_curved.argtypes = [ctypes.c_void_p, ctypes.c_int]
            set_screen_curved.restype = ctypes.c_int
            self._screen_curved_abi_available = True
        set_screen_light = getattr(
            library, "filament_bridge_set_screen_light", None
        )
        if set_screen_light is not None:
            set_screen_light.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float,
            ]
            set_screen_light.restype = ctypes.c_int
            self._screen_light_abi_available = True
        set_glow_source = getattr(
            library, "filament_bridge_set_glow_source", None
        )
        set_glow_state = getattr(
            library, "filament_bridge_set_glow_state", None
        )
        set_glow_image = getattr(
            library, "filament_bridge_set_glow_image", None
        )
        if set_glow_source is not None and set_glow_state is not None:
            set_glow_source.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            set_glow_source.restype = ctypes.c_int
            set_glow_state.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                *([ctypes.c_float] * 19),
            ]
            set_glow_state.restype = ctypes.c_int
            self._glow_abi_available = True
        if set_glow_image is not None:
            set_glow_image.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_int32,
            ]
            set_glow_image.restype = ctypes.c_int
            self._glow_vulkan_image_abi_available = True
        set_screen_sampling = getattr(
            library, "filament_bridge_set_screen_sampling", None
        )
        if set_screen_sampling is not None:
            set_screen_sampling.argtypes = [ctypes.c_void_p, ctypes.c_float]
            set_screen_sampling.restype = ctypes.c_int
            self._screen_sampling_abi_available = True
        set_screen_upscale = getattr(
            library, "filament_bridge_set_screen_upscale", None
        )
        if set_screen_upscale is not None:
            set_screen_upscale.argtypes = [ctypes.c_void_p, ctypes.c_float]
            set_screen_upscale.restype = ctypes.c_int
            self._screen_upscale_abi_available = True
        set_screen_sampling_mode = getattr(
            library, "filament_bridge_set_screen_sampling_mode", None
        )
        if set_screen_sampling_mode is not None:
            set_screen_sampling_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            set_screen_sampling_mode.restype = ctypes.c_int
            self._screen_sampling_mode_abi_available = True
        get_screen_sampling_stats = getattr(
            library, "filament_bridge_get_screen_sampling_stats", None
        )
        if get_screen_sampling_stats is not None:
            get_screen_sampling_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            get_screen_sampling_stats.restype = ctypes.c_int
            self._screen_sampling_stats_abi_available = True
        if hasattr(library, "filament_bridge_set_screen_image"):
            library.filament_bridge_set_screen_image.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_int32,
            ]
            library.filament_bridge_set_screen_image.restype = ctypes.c_int
            self._screen_image_abi_available = True
        set_screen_source_version = getattr(
            library, "filament_bridge_set_screen_source_version", None
        )
        if set_screen_source_version is not None:
            set_screen_source_version.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
            ]
            set_screen_source_version.restype = ctypes.c_int
            self._screen_source_version_abi_available = True
        fixed_screen_image = getattr(
            library, "filament_bridge_set_fixed_screen_image", None
        )
        if fixed_screen_image is not None:
            fixed_screen_image.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            fixed_screen_image.restype = ctypes.c_int
            self._fixed_screen_image_abi_available = True
        external_image_abi = getattr(
            library, "filament_bridge_vulkan_external_image_abi_available", None
        )
        if external_image_abi is not None:
            external_image_abi.argtypes = [ctypes.c_void_p]
            external_image_abi.restype = ctypes.c_int
            self._vulkan_external_image_abi_available = bool(
                external_image_abi(None)
            )
        self._glow_vulkan_image_abi_available = bool(
            self._glow_vulkan_image_abi_available
            and self._vulkan_external_image_abi_available
        )
        if hasattr(library, "filament_bridge_set_screen_ready_semaphore"):
            library.filament_bridge_set_screen_ready_semaphore.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p
            ]
            library.filament_bridge_set_screen_ready_semaphore.restype = ctypes.c_int
            self._screen_ready_semaphore_abi_available = True
        if hasattr(library, "filament_bridge_get_finished_drawing_semaphore"):
            library.filament_bridge_get_finished_drawing_semaphore.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            ]
            library.filament_bridge_get_finished_drawing_semaphore.restype = ctypes.c_int
            self._finished_drawing_semaphore_abi_available = True
        library.filament_bridge_apply_animations.argtypes = [
            ctypes.c_void_p, ctypes.c_double
        ]
        library.filament_bridge_apply_animations.restype = ctypes.c_int
        library.filament_bridge_last_error.argtypes = [ctypes.c_void_p]
        library.filament_bridge_last_error.restype = ctypes.c_char_p

    def _ensure_controller_abi(self) -> None:
        if not self._controller_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge controller ABI is unavailable; rebuild the CI artifact"
            )

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            raise FilamentBridgeError("Filament Bridge is not initialized")
        self._ensure_owner_thread()

    def _ensure_owner_thread(self) -> None:
        if (
            self._owner_thread_id is not None
            and threading.get_ident() != self._owner_thread_id
        ):
            raise FilamentBridgeError(
                "Filament Bridge C ABI must run on its Presenter owner thread"
            )

    def _raise_if_error(self) -> None:
        message = self._last_error()
        if message:
            self.close()
            raise FilamentBridgeError(f"create_vulkan: {message}")

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


def _array_payload(value: Any) -> bytes:
    """Serialize NumPy buffers safely, including zero-length arrays."""
    tobytes = getattr(value, "tobytes", None)
    if tobytes is not None:
        return bytes(tobytes(order="C"))
    return bytes(memoryview(value).cast("B"))
