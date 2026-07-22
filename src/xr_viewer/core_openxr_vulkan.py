from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from viewer.vulkan_context import (
    MIN_VULKAN_API_VERSION,
    VulkanContext,
    VulkanCapabilityError,
    _require_timeline_semaphore_features,
    find_graphics_queue_family,
    make_vulkan_version,
)
from viewer.vulkan_resources import VulkanExportableImage, VulkanHostImage, VulkanImageResource
from app_runtime.output_contract import VulkanStereoOutputFrame


_OUTPUT_FRAME_UNSET = object()

from .core_controller_actions import CoreControllerActionsMixin
from .core_input_helpers import CoreInputHelpersMixin
from .core_controller_input import CoreControllerInputMixin
from .core_controller_pose import CoreControllerPoseMixin
from .core_controller_ray import CoreControllerRayMixin
from .controller_models import discover_controller_brands, select_controller_brand
from .filters import OneEuroFilter3D
from .xr_math import (
    _mat3_to_quat_xyzw,
    _xr_quat_to_mat4,
    euler_to_mat4,
    mat4_to_xr_posef,
)
from .overlay_textures import (
    build_cursor_rgba,
    build_fps_overlay_rgba,
    build_help_rgba,
    build_keyboard_rgba,
    build_short_osd_rgba,
)
from .windows_input import (
    _MOUSEEVENTF_LEFTDOWN,
    _MOUSEEVENTF_LEFTUP,
    _MOUSEEVENTF_RIGHTDOWN,
    _MOUSEEVENTF_RIGHTUP,
    _send_mouse_flags,
    _send_key,
    _set_cursor_pos,
    _get_desktop_size,
)


class OpenXrVulkanUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenXrVulkanConfig:
    application_name: str = "Desktop2Stereo Vulkan"
    render_scale: float = 1.0
    clear_color: tuple[float, float, float, float] = (0.02, 0.04, 0.08, 1.0)
    requested_vulkan_version: int = make_vulkan_version(1, 4, 0)
    # Keep the validated OpenXR projection target as sRGB. The Filament bridge
    # is configured for linear Rec709 output so the target performs one OETF.
    swapchain_color_mode: str = "srgb"
    controller_model: str = "PICO"
    filament_bridge_path: str | None = None
    filament_glb_path: str | None = None
    filament_profile_path: str | None = None
    filament_scene_exposure_ev: float = 0.0
    filament_skybox_brightness: float = 1.0
    filament_fill_light_color: tuple[float, float, float] = (1.0, 0.88, 0.78)
    filament_fill_light_intensity: float = 100000.0
    filament_fill_light_direction: tuple[float, float, float] = (-0.35, -1.0, -0.55)
    openxr_no_headset_retry_interval: float = 3.0
    openxr_standby_retry_interval: float = 3.0
    openxr_standby_retry_max_interval: float = 30.0
    headset_wait_inference_timeout: float = 60.0


@dataclass(slots=True)
class _EyeSwapchain:
    handle: Any
    images: list[Any]
    width: int
    height: int
    resources: list[VulkanImageResource] = field(default_factory=list)


class OpenXrCompositionBuilder:
    """Builds projection layers without owning OpenXR frame lifecycle."""

    def __init__(self, xr: Any, reference_space: Any) -> None:
        self.xr = xr
        self.reference_space = reference_space

    def projection_layer(
        self, views: list[Any], swapchains: list[_EyeSwapchain]
    ) -> Any:
        if len(views) < len(swapchains):
            raise ValueError("projection layer requires one view per eye swapchain")
        projection_views = []
        for eye_index, eye in enumerate(swapchains):
            projection_views.append(
                self.xr.CompositionLayerProjectionView(
                    pose=views[eye_index].pose,
                    fov=views[eye_index].fov,
                    sub_image=self.xr.SwapchainSubImage(
                        swapchain=eye.handle,
                        image_rect=self.xr.Rect2Di(
                            offset=self.xr.Offset2Di(x=0, y=0),
                            extent=self.xr.Extent2Di(width=eye.width, height=eye.height),
                        ),
                        image_array_index=0,
                    ),
                )
            )
        return self.xr.CompositionLayerProjection(
            space=self.reference_space,
            views=projection_views,
        )

    def quad_layer(
        self, swapchain: _EyeSwapchain, position: tuple[float, float, float],
        width: float, height: float, rotation: tuple[float, float, float],
        eye_index: int,
    ) -> Any:
        qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            space=self.reference_space,
            eye_visibility=(self.xr.EyeVisibility.LEFT if eye_index == 0
                            else self.xr.EyeVisibility.RIGHT),
            sub_image=self.xr.SwapchainSubImage(
                swapchain=swapchain.handle,
                image_rect=self.xr.Rect2Di(
                    offset=self.xr.Offset2Di(x=0, y=0),
                    extent=self.xr.Extent2Di(width=swapchain.width, height=swapchain.height),
                ),
                image_array_index=0,
            ),
            pose=self.xr.Posef(
                orientation=self.xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=self.xr.Vector3f(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                ),
            ),
            size=self.xr.Extent2Df(width=float(width), height=float(height)),
        )


class OpenXrVulkanPresenter(
    CoreControllerActionsMixin,
    CoreControllerPoseMixin,
    CoreControllerRayMixin,
    CoreControllerInputMixin,
    CoreInputHelpersMixin,
):
    """OpenXR Vulkan projection-layer presenter with Filament controllers."""

    _VULKAN_EXTENSION = "XR_KHR_vulkan_enable2"

    def __init__(
        self,
        config: OpenXrVulkanConfig | None = None,
        *,
        on_headset_state: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or OpenXrVulkanConfig()
        self._on_headset_state = on_headset_state
        if self.config.render_scale <= 0:
            raise ValueError("render_scale must be greater than zero")
        if len(self.config.clear_color) != 4:
            raise ValueError("clear_color must contain four components")

        self.xr: Any = None
        self.instance: Any = None
        self.system_id: Any = None
        self.session: Any = None
        self.reference_space: Any = None
        self._reference_space_type: Any = None
        self.vulkan: VulkanContext | None = None
        self.swapchain_format: int | None = None
        self.swapchains: list[_EyeSwapchain] = []
        self._quad_swapchains: list[_EyeSwapchain] = []
        self._quad_swapchain_format: int | None = None
        self._quad_swapchain_extent: tuple[int, int] | None = None
        self.filament_bridge: Any | None = None
        self.session_state: Any = None
        self.session_running = False
        self.exit_requested = False
        self.frame_count = 0
        self._view_configuration_type: Any = None
        self._environment_blend_mode: Any = None
        self._vulkan_loader: Any = None
        self._vk_get_instance_proc_addr: Any = None
        self._graphics_binding: Any = None
        self._provisional_vk_instance: Any = None
        self._provisional_vk_device: Any = None
        self._profile_head_transform: np.ndarray | None = None
        self._profile_initial_head: np.ndarray | None = None
        self._profile_space_applied = False
        self._profile_view_name: str | None = None
        self._profile_near_plane = 0.05
        self._profile_far_plane = 1000.0
        self._filament_scene_exposure = self.config.filament_scene_exposure_ev
        self._filament_skybox_brightness = self.config.filament_skybox_brightness
        self._filament_fill_light_color = self.config.filament_fill_light_color
        self._filament_fill_light_intensity = self.config.filament_fill_light_intensity
        self._filament_fill_light_direction = self.config.filament_fill_light_direction
        self._filament_screen: tuple[
            tuple[float, float, float], float, float, tuple[float, float, float]
        ] | None = None
        self._filament_screen_initial = None
        # The virtual screen must be rendered as scene geometry so each eye
        # samples its own stereo output. Quad Layer is reserved for 2D tools.
        self._filament_screen_image_enabled = os.environ.get(
            # Prefer zero-copy when the per-frame synchronization contract is
            # available; _can_use_filament_screen_image provides the fallback.
            "D2S_ENABLE_FILAMENT_SCREEN_IMAGE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._controllers_root = Path(__file__).resolve().parent / "controllers"
        self._controller_brands = discover_controller_brands(self._controllers_root)
        self._controller_brand = select_controller_brand(
            self._controller_brands,
            self.config.controller_model or os.environ.get("D2S_CONTROLLER_MODEL", "PICO"),
        )
        self._controller_inputs = ({}, {})
        self._aim_space_l = None
        self._aim_space_r = None
        self._grip_space_l = None
        self._grip_space_r = None
        self._aim_mat_l = None
        self._aim_mat_r = None
        self._grip_mat_l = None
        self._grip_mat_r = None
        self._frame_now = 0.0
        self._filament_animation_origin: float | None = None
        # Keep the controller lifecycle aligned with the legacy renderer:
        # movement refreshes a per-hand activity timestamp and both the model
        # and laser are hidden after the idle timeout.
        controller_now = time.perf_counter()
        self._laser_last_move_l = controller_now
        self._laser_last_move_r = controller_now
        self._laser_prev_mat_l = None
        self._laser_prev_mat_r = None
        self._LASER_HIDE_AFTER = 5.0
        self._LASER_MOVE_THRESH = 0.015
        self._smooth_ray_origin_l = None
        self._smooth_ray_origin_r = None
        self._smooth_ray_quat_l = None
        self._smooth_ray_quat_r = None
        self._smooth_ray_fwd_l = None
        self._smooth_ray_fwd_r = None
        self._rot_smooth = 0.10
        self._ray_deadzone_rad = 0.0052
        self._ray_filter_l = OneEuroFilter3D(8.0, 8.0, 8.0)
        self._ray_filter_r = OneEuroFilter3D(8.0, 8.0, 8.0)
        self._last_frame_dt = 1.0 / 90.0
        self._initialized = False
        self._pending_output: VulkanStereoOutputFrame | None = None
        self._displayed_output: VulkanStereoOutputFrame | None = None
        self._rendering_output: VulkanStereoOutputFrame | None = None
        self._output_lock = threading.Lock()
        self._headset_wait_started = 0.0
        self._headset_hard_idle_notified = False
        self._headset_active_notified = False
        self._headset_wait_logged = False
        self._accept_output = False
        self._source_frame_wait_logged = False
        self._has_presented_frame = False
        self._last_quad_layers: list[Any] = []
        self._overlay_quad_entries: dict[str, dict[str, Any]] = {}
        # Legacy OpenXR shortcut state is kept in the presenter so both the
        # Vulkan projection path and future Quad Layer overlays read one state.
        self._keyboard_visible = False
        self._fps_overlay_visible = False
        self._operation_guide_visible = False
        self._aperture_visible = False
        self._shortcut_last = {}
        self._keyboard_width = 1.6
        self._keyboard_height = 0.33
        self._keyboard_keys = []
        self._kb_show_shifted = False
        self._mod_state = {
            "shift": [False, False, 0.0],
            "ctrl": [False, False, 0.0],
            "alt": [False, False, 0.0],
            "win": [False, False, 0.0],
        }
        self._caps_lock = False
        self._kb_trig_prev_l = 0.0
        self._kb_trig_prev_r = 0.0
        self._kb_hover_l = None
        self._kb_hover_r = None
        self._kb_held_key_l = None
        self._kb_held_key_r = None
        self._kb_held_mods_l = None
        self._kb_held_mods_r = None
        self._grip_l_now = False
        self._grip_r_now = False
        self._pointer_state = {"left": "idle", "right": "idle"}
        self._pointer_press_time = {"left": 0.0, "right": 0.0}
        self._left_grab_anchor = None
        self._right_grab_anchor = None
        self._keyboard_position_offset = np.zeros(3, dtype=np.float64)
        self._keyboard_grab_anchor = None
        self._screen_resize_anchor = None
        self._menu_pressed_last = False
        self._menu_press_time = 0.0
        self._menu_long_fired = False
        self._button_press_time = {"a": 0.0, "b": 0.0, "x": 0.0, "y": 0.0}
        self._button_long_fired = {"a": False, "b": False, "x": False, "y": False}
        self._stick_click_last = {"left": False, "right": False}
        self._stick_press_time = {"left": 0.0, "right": 0.0}
        self._stick_long_fired = {"left": False, "right": False}

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        self.exit_requested = False
        self.frame_count = 0
        self.session_state = None
        self.xr = _import_openxr()
        xr = self.xr
        available_extensions = {
            _decode_name(item.extension_name)
            for item in xr.enumerate_instance_extension_properties()
        }
        if self._VULKAN_EXTENSION not in available_extensions:
            raise OpenXrVulkanUnavailableError(
                f"OpenXR runtime does not expose {self._VULKAN_EXTENSION}"
            )

        try:
            self.instance = xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(
                        application_name=self.config.application_name,
                        application_version=1,
                        engine_name="D2S",
                        engine_version=1,
                        api_version=xr.Version(1, 0, 0),
                    ),
                    enabled_extension_names=[self._VULKAN_EXTENSION],
                )
            )
            self.system_id = xr.get_system(
                self.instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )
            requirements = _get_vulkan_graphics_requirements2(
                xr, self.instance, self.system_id
            )
            api_version = _select_vulkan_api_version(
                requirements, self.config.requested_vulkan_version
            )
            self._create_vulkan_objects(api_version)
            self._create_session_and_swapchains()
            self._xr_instance = self.instance
            self._xr_session = self.session
            self._xr_space = self.reference_space
            self._init_controller_actions()
            self._load_filament_profile()
            self._initialize_filament_bridges()
            self._initialized = True
        except Exception:
            self.close()
            raise

    def poll_events(self) -> None:
        self._ensure_initialized()
        xr = self.xr
        while True:
            try:
                event = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return

            if event.type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                changed = ctypes.cast(
                    ctypes.byref(event),
                    ctypes.POINTER(xr.EventDataSessionStateChanged),
                ).contents
                self.session_state = changed.state
                if changed.state == xr.SessionState.READY and not self.session_running:
                    xr.begin_session(
                        self.session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=self._view_configuration_type
                        ),
                    )
                    self.session_running = True
                elif changed.state == xr.SessionState.STOPPING and self.session_running:
                    xr.end_session(self.session)
                    self.session_running = False
                elif changed.state in (
                    xr.SessionState.EXITING,
                    xr.SessionState.LOSS_PENDING,
                ):
                    self.exit_requested = True
            elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self.exit_requested = True

    def run_frame(self) -> bool:
        self._ensure_initialized()
        self.poll_events()
        if self.exit_requested:
            return False
        if not self.session_running:
            self._notify_headset_waiting()
            time.sleep(0.01)
            return True

        xr = self.xr
        frame_state = xr.wait_frame(self.session)
        previous_frame_now = self._frame_now
        self._frame_now = time.perf_counter()
        if previous_frame_now > 0.0:
            self._last_frame_dt = max(
                0.001, min(0.1, self._frame_now - previous_frame_now)
            )
        if frame_state.should_render:
            self._notify_headset_active()
        else:
            self._notify_headset_waiting()
        try:
            self._sync_controller_inputs(1.0 / 90.0)
            self._handle_controller_shortcuts()
            self._update_aim_poses(frame_state.predicted_display_time)
            self._update_grip_poses(frame_state.predicted_display_time)
            self._smooth_controller_poses()
            self._grip_l_now = bool(self._controller_input(0).get("grip", 0.0) > 0.5)
            self._grip_r_now = bool(self._controller_input(1).get("grip", 0.0) > 0.5)
            self._handle_keyboard_input()
            self._handle_vulkan_pointer_input()
        except Exception:
            pass
        xr.begin_frame(self.session)
        layer_structures: list[Any] = []
        layer_pointers: list[Any] = []
        try:
            if frame_state.should_render:
                view_state, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(
                        view_configuration_type=self._view_configuration_type,
                        display_time=frame_state.predicted_display_time,
                        space=self.reference_space,
                    ),
                )
                valid_flags = (
                    xr.ViewStateFlags.POSITION_VALID_BIT
                    | xr.ViewStateFlags.ORIENTATION_VALID_BIT
                )
                if view_state.view_state_flags & valid_flags == valid_flags:
                    if self._apply_profile_reference_space(views):
                        view_state, views = xr.locate_views(
                            self.session,
                            xr.ViewLocateInfo(
                                view_configuration_type=self._view_configuration_type,
                                display_time=frame_state.predicted_display_time,
                                space=self.reference_space,
                            ),
                        )
                    with self._output_lock:
                        output_frame = self._pending_output
                    # Match the legacy frame gate: runtime rendering readiness
                    # is separate from the availability of a fresh stereo frame.
                    if self._pending_output is None and not self._has_presented_frame:
                        if not self._source_frame_wait_logged:
                            self._source_frame_wait_logged = True
                            print(
                                "[OpenXRViewer] OpenXR render ready; "
                                "waiting for first runtime eye frame",
                                flush=True,
                            )
                        layer = None
                    else:
                        self._source_frame_wait_logged = False
                        # Render the world at the current headset pose on
                        # every XR tick; only inference input may be reused.
                        layer = self._render_projection_layer(views, output_frame)
                    if layer is not None:
                        layer_structures.append(layer)
                        layer_pointers.append(ctypes.pointer(layer))
                        if output_frame is not None:
                            try:
                                quad_layers = self._render_quad_layers(output_frame)
                                if quad_layers:
                                    self._last_quad_layers = quad_layers
                                self._commit_output_frame(output_frame)
                            except Exception:
                                self._abort_output_frame(output_frame)
                                raise
                        self._has_presented_frame = True
                        layer_structures.extend(self._last_quad_layers)
                        layer_pointers.extend(
                            ctypes.pointer(item) for item in self._last_quad_layers
                        )
        finally:
            end_info = xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=self._environment_blend_mode,
                layer_count=len(layer_pointers),
                layers=layer_pointers or None,
            )
            xr.end_frame(self.session, end_info)
        self.frame_count += 1
        return not self.exit_requested

    def _handle_controller_shortcuts(self) -> None:
        """Apply the legacy short/long press state machine to Vulkan state."""
        left, right = self._controller_inputs
        now = time.perf_counter()

        def pressed(hand: dict[str, float], name: str) -> bool:
            return float(hand.get(name, 0.0) or 0.0) > 0.5

        def set_panel(name: str | None) -> None:
            self._fps_overlay_visible = name == "fps"
            self._operation_guide_visible = name == "guide"
            self._aperture_visible = name == "aperture"

        def cycle_panel() -> None:
            current = (
                "fps" if self._fps_overlay_visible else
                "guide" if self._operation_guide_visible else
                "aperture" if self._aperture_visible else None
            )
            order = [None, "fps", "guide", "aperture"]
            set_panel(order[(order.index(current) + 1) % len(order)])

        menu_now = pressed(left, "menu_button") or pressed(right, "menu_button")
        if menu_now and not self._menu_pressed_last:
            self._menu_press_time = now
            self._menu_long_fired = False
        if menu_now and not self._menu_long_fired and now - self._menu_press_time >= 0.6:
            self._menu_long_fired = True
            set_panel("aperture")
        if not menu_now and self._menu_pressed_last and not self._menu_long_fired:
            cycle_panel()
        self._menu_pressed_last = menu_now

        def button(name: str, value: bool, short_action, long_action=None) -> None:
            if value and not self._shortcut_last.get(name, False):
                self._button_press_time[name] = now
                self._button_long_fired[name] = False
            if value and not self._button_long_fired[name] and long_action is not None:
                if now - self._button_press_time[name] >= 1.0:
                    long_action()
                    self._button_long_fired[name] = True
            if not value and self._shortcut_last.get(name, False) and not self._button_long_fired[name]:
                short_action()
            self._shortcut_last[name] = value

        def toggle_keyboard() -> None:
            self._keyboard_visible = not self._keyboard_visible
            self._keyboard_position_offset[:] = 0.0
            self._keyboard_grab_anchor = None

        def reset_screen() -> None:
            if self._filament_screen_initial is not None:
                self._set_filament_screen_pose(
                    self._filament_screen_initial[0], self._filament_screen_initial[3]
                )

        button("x", pressed(left, "x_button"), toggle_keyboard, lambda: set_panel("fps"))
        button("a", pressed(right, "a_button"), lambda: set_panel(None if self._operation_guide_visible else "guide"), lambda: set_panel("fps"))
        button("b", pressed(right, "b_button"), lambda: set_panel(None if self._aperture_visible else "aperture"), lambda: set_panel("aperture"))
        button("y", pressed(left, "y_button"), reset_screen, cycle_panel)

        for hand_name, hand in (("left", left), ("right", right)):
            value = pressed(hand, "stick_click")
            was_down = self._stick_click_last[hand_name]
            if value and not was_down:
                self._stick_press_time[hand_name] = now
                self._stick_long_fired[hand_name] = False
            if value and not self._stick_long_fired[hand_name] and now - self._stick_press_time[hand_name] >= 1.0:
                _send_key(0x58 if hand_name == "left" else 0x0D, ctrl=(hand_name == "left"))
                self._stick_long_fired[hand_name] = True
            if not value and was_down and not self._stick_long_fired[hand_name]:
                _send_key(0x43 if hand_name == "left" else 0x56, ctrl=True)
            self._stick_click_last[hand_name] = value

    def _input_deadzone(self) -> float:
        return 0.15

    def _pulse_haptic(self, *args, **kwargs) -> None:
        # Haptics are optional in the Vulkan migration; keyboard input remains
        # independent of runtime-specific vibration support.
        return None

    def _press_key(self, key, key_idx, held_key_attr, held_mods_attr):
        return self._press_key_impl(key, key_idx, held_key_attr, held_mods_attr)

    def _refresh_or_upload_keyboard_content(self) -> None:
        # Tool quads rebuild their RGBA payload from the current state each XR tick.
        return None

    def _adjust_frosted_glow_vk(self, _vk_code: int) -> bool:
        return False

    def _keyboard_pose_mat4(self) -> np.ndarray:
        position, screen_width, screen_height, rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        matrix = euler_to_mat4(*(math.radians(float(value)) for value in rotation))
        matrix[:3, 3] = np.asarray(
            (position[0], position[1] - screen_height * 0.72, position[2]),
            dtype=np.float64,
        ) + self._keyboard_position_offset
        return matrix.astype(np.float64)

    def _keyboard_plane_hit(self, origin, direction):
        if not self._keyboard_visible:
            return None, None
        if not self._keyboard_keys:
            _rgba, self._keyboard_keys = build_keyboard_rgba(
                self._kb_show_shifted, self._keyboard_width, self._keyboard_height
            )
        pose = self._keyboard_pose_mat4()
        normal = pose[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None, None
        distance = float(np.dot(normal, pose[:3, 3] - origin) / denominator)
        if distance <= 0.0:
            return None, None
        hit = np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * distance
        local = np.linalg.inv(pose) @ np.append(hit, 1.0)
        x, y = float(local[0]), float(local[1])
        if abs(x) > self._keyboard_width / 2.0 or abs(y) > self._keyboard_height / 2.0:
            return None, None
        return x, y

    def _screen_ray_hit(self, matrix):
        if matrix is None or self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(*(math.radians(float(value)) for value in rotation)).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        origin = matrix[:3, 3].astype(np.float64)
        direction = (-matrix[:3, 2]).astype(np.float64)
        normal = pose[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None
        distance = float(np.dot(normal, pose[:3, 3] - origin) / denominator)
        if distance <= 0.0:
            return None
        hit = origin + direction * distance
        local = np.linalg.inv(pose) @ np.append(hit, 1.0)
        if abs(float(local[0])) > width / 2.0 or abs(float(local[1])) > height / 2.0:
            return None
        return (
            max(0.0, min(1.0, float(local[0]) / width + 0.5)),
            max(0.0, min(1.0, 0.5 - float(local[1]) / height)),
        )

    def _set_filament_screen_pose(self, position, rotation=None) -> None:
        if self._filament_screen is None:
            return
        _old_position, width, height, old_rotation = self._filament_screen
        pose_rotation = tuple(rotation if rotation is not None else old_rotation)
        self._filament_screen = (tuple(float(value) for value in position), width, height, pose_rotation)
        if self.filament_bridge is not None:
            self.filament_bridge.set_screen(self._filament_screen[0], width, height, pose_rotation)

    def _can_use_filament_screen_image(
        self, output_frame: VulkanStereoOutputFrame | None
    ) -> bool:
        """Require a producer completion primitive before zero-copy sampling.

        A raw VkImage handle is not sufficient to establish visibility for a
        Filament shader. The producer must publish both per-eye ready
        semaphores, and the bridge must consume them from the same Vulkan
        device before rendering the imported image.
        """
        if (
            output_frame is None
            or not self._filament_screen_image_enabled
            or self.filament_bridge is None
            or not getattr(self.filament_bridge, "screen_image_abi_available", False)
            or not getattr(
                self.filament_bridge, "screen_ready_semaphore_abi_available", False
            )
        ):
            return False
        metadata = dict(output_frame.metadata or {})
        if metadata.get("vulkan_output_sync") != "cuda_external_semaphore":
            return False
        return bool(
            metadata.get("vulkan_ready_semaphore_left")
            and metadata.get("vulkan_ready_semaphore_right")
        )

    def _handle_vulkan_pointer_input(self) -> None:
        """Reuse legacy trigger hold/drag semantics for the Vulkan screen."""
        now = time.perf_counter()
        inputs = (self._controller_input(0), self._controller_input(1))
        hits = (self._screen_ray_hit(self._aim_mat_l), self._screen_ray_hit(self._aim_mat_r))
        left_grip = bool(inputs[0].get("grip", 0.0) > 0.5)
        right_grip = bool(inputs[1].get("grip", 0.0) > 0.5)
        if left_grip and self._grip_mat_l is not None:
            grip_position = self._grip_mat_l[:3, 3].astype(np.float64)
            if self._keyboard_visible and self._keyboard_plane_hit(
                self._grip_mat_l[:3, 3], -self._grip_mat_l[:3, 2]
            ) != (None, None):
                if self._keyboard_grab_anchor is None:
                    self._keyboard_grab_anchor = self._keyboard_pose_mat4()[:3, 3] - grip_position
                keyboard_position = grip_position + self._keyboard_grab_anchor
                screen_position = np.asarray(self._filament_screen[0], dtype=np.float64)
                self._keyboard_position_offset = keyboard_position - np.asarray(
                    (screen_position[0], screen_position[1] - self._filament_screen[2] * 0.72, screen_position[2]),
                    dtype=np.float64,
                )
            elif hits[0] is not None:
                if self._left_grab_anchor is None and self._filament_screen is not None:
                    self._left_grab_anchor = np.asarray(self._filament_screen[0], dtype=np.float64) - grip_position
                if self._left_grab_anchor is not None:
                    self._set_filament_screen_pose(grip_position + self._left_grab_anchor)
        else:
            self._left_grab_anchor = None
            self._keyboard_grab_anchor = None
        if right_grip and self._grip_mat_r is not None and self._filament_screen is not None:
            grip_position = self._grip_mat_r[:3, 3].astype(np.float64)
            if self._screen_resize_anchor is None:
                self._screen_resize_anchor = (grip_position.copy(), float(self._filament_screen[1]))
            delta = float(grip_position[0] - self._screen_resize_anchor[0][0])
            new_width = max(0.3, min(20.0, self._screen_resize_anchor[1] + delta * 1.5))
            old_position, _old_width, old_height, old_rotation = self._filament_screen
            self._filament_screen = (
                old_position,
                new_width,
                new_width * float(old_height) / max(float(_old_width), 1e-6),
                old_rotation,
            )
            if self.filament_bridge is not None:
                self.filament_bridge.set_screen(
                    old_position, self._filament_screen[1], self._filament_screen[2], old_rotation
                )
        else:
            self._screen_resize_anchor = None
        for name, hand, hit, down_flag, up_flag in (
            ("left", inputs[0], hits[0], _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
            ("right", inputs[1], hits[1], _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
        ):
            trigger = float(hand.get("trigger", 0.0) or 0.0)
            state = self._pointer_state[name]
            aim_matrix = self._aim_mat_l if name == "left" else self._aim_mat_r
            keyboard_hit = False
            if self._keyboard_visible and aim_matrix is not None:
                keyboard_hit = self._keyboard_plane_hit(
                    aim_matrix[:3, 3], -aim_matrix[:3, 2]
                ) != (None, None)
            if hit is None or keyboard_hit:
                if state != "idle":
                    _send_mouse_flags(up_flag)
                self._pointer_state[name] = "idle"
                continue
            if state == "idle" and trigger >= 0.7:
                _set_cursor_pos(int(hit[0] * _get_desktop_size()[0]), int(hit[1] * _get_desktop_size()[1]))
                _send_mouse_flags(down_flag)
                _send_mouse_flags(up_flag)
                self._pointer_press_time[name] = now
                self._pointer_state[name] = "pressed"
            elif state == "pressed":
                if trigger <= 0.3:
                    self._pointer_state[name] = "idle"
                elif now - self._pointer_press_time[name] >= 0.35:
                    _send_mouse_flags(down_flag)
                    self._pointer_state[name] = "dragging"
            elif state == "dragging":
                if trigger <= 0.3:
                    _send_mouse_flags(up_flag)
                    self._pointer_state[name] = "idle"
                else:
                    _set_cursor_pos(int(hit[0] * _get_desktop_size()[0]), int(hit[1] * _get_desktop_size()[1]))

    def run(self, frame_limit: int | None = None) -> int:
        self.initialize()
        while frame_limit is None or self.frame_count < frame_limit:
            if not self.run_frame():
                break
        return self.frame_count

    def run_until(self, shutdown_event: Any) -> int:
        """Run the XR frame loop until the application shutdown event is set."""
        retry_count = 0
        try:
            while not shutdown_event.is_set() and not self.exit_requested:
                try:
                    if not self._initialized:
                        self.initialize()
                    retry_count = 0
                    while not shutdown_event.is_set() and not self.exit_requested:
                        if not self.run_frame():
                            break
                        if self._session_requires_reconnect():
                            self.close()
                            self.exit_requested = False
                            self._notify_headset_waiting()
                            break
                    if self._session_requires_reconnect():
                        self.close()
                        self.exit_requested = False
                        self._notify_headset_waiting()
                except Exception as exc:
                    if not self._is_no_headset_error(exc):
                        raise
                    print(
                        "[OpenXRViewer] OpenXR HMD form factor unavailable; "
                        "Vulkan/Filament initialization deferred until headset wake-up",
                        flush=True,
                    )
                    self.close()
                    self._notify_headset_waiting()

                if shutdown_event.is_set() or self.exit_requested:
                    break
                retry_count += 1
                delay = self._retry_delay(retry_count)
                print(
                    f"[OpenXRViewer] Waiting for VR headset connect... "
                    f"(retry in {delay:.1f}s)",
                    flush=True,
                )
                shutdown_event.wait(delay)
            return self.frame_count
        finally:
            self.close()

    @staticmethod
    def _is_no_headset_error(exc: BaseException) -> bool:
        return type(exc).__name__ == "FormFactorUnavailableError"

    def _session_requires_reconnect(self) -> bool:
        state = self.session_state
        state_name = str(getattr(state, "name", state)).upper()
        return state_name in {"STOPPING", "LOSS_PENDING"}

    def _retry_delay(self, retry_count: int) -> float:
        base = max(0.1, float(self.config.openxr_standby_retry_interval))
        maximum = max(base, float(self.config.openxr_standby_retry_max_interval))
        if self.session_state is None:
            base = max(0.1, float(self.config.openxr_no_headset_retry_interval))
        return min(maximum, base * (2 ** max(0, retry_count - 1)))

    def _notify_headset_state(self, state: str) -> None:
        callback = self._on_headset_state
        if callback is None:
            return
        try:
            callback(state)
        except Exception as exc:
            print(
                f"[OpenXRViewer] Headset state callback failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _notify_headset_waiting(self) -> None:
        # Do not let a frame produced before standby cross the recovery boundary.
        self._accept_output = False
        self._drop_output_frames()
        now = time.perf_counter()
        if self._headset_wait_started <= 0.0:
            self._headset_wait_started = now
            self._headset_hard_idle_notified = False
            self._headset_active_notified = False
            self._headset_wait_logged = False
            self._notify_headset_state("waiting")
        if not self._headset_wait_logged:
            self._headset_wait_logged = True
            print(
                "[OpenXRViewer] Headset not detected or in standby; "
                "waiting for headset wake-up",
                flush=True,
            )
        timeout = max(0.0, float(self.config.headset_wait_inference_timeout))
        if (
            not self._headset_hard_idle_notified
            and now - self._headset_wait_started >= timeout
        ):
            self._headset_hard_idle_notified = True
            self._notify_headset_state("hard_idle")
            print(
                f"[OpenXRViewer] No headset detected for {timeout:.0f}s; "
                "stopping source inference",
                flush=True,
            )

    def _notify_headset_active(self) -> None:
        if self._headset_active_notified:
            return
        self._headset_wait_started = 0.0
        self._headset_hard_idle_notified = False
        self._headset_active_notified = True
        self._headset_wait_logged = False
        self._source_frame_wait_logged = False
        self._accept_output = True
        self._notify_headset_state("active")
        print("[OpenXRViewer] Headset detected; source inference resumed", flush=True)

    def close(self) -> None:
        xr = self.xr
        if self.vulkan is not None:
            try:
                self.vulkan.wait_idle()
            except Exception:
                pass

        self._drop_output_frames()

        if self.filament_bridge is not None:
            try:
                self.filament_bridge.close()
            except Exception:
                pass
            self.filament_bridge = None

        if xr is not None:
            self._destroy_tool_quad_layers()
            self._destroy_quad_swapchains()
            for eye in reversed(self.swapchains):
                for resource in reversed(eye.resources):
                    try:
                        if self.vulkan is not None:
                            self.vulkan.unregister_external_image(resource)
                    except Exception:
                        pass
                try:
                    xr.destroy_swapchain(eye.handle)
                except Exception:
                    pass
            self.swapchains.clear()

            if self.reference_space is not None:
                try:
                    xr.destroy_space(self.reference_space)
                except Exception:
                    pass
                self.reference_space = None

            if self.session is not None:
                if self.session_running:
                    try:
                        xr.end_session(self.session)
                    except Exception:
                        pass
                try:
                    xr.destroy_session(self.session)
                except Exception:
                    pass
                self.session = None
                self.session_running = False

        if self.vulkan is not None:
            try:
                self.vulkan.close()
            except Exception:
                pass
            self.vulkan = None
        elif self._provisional_vk_instance is not None:
            try:
                import vulkan as vk

                if self._provisional_vk_device is not None:
                    vk.vkDestroyDevice(self._provisional_vk_device, None)
                vk.vkDestroyInstance(self._provisional_vk_instance, None)
            except Exception:
                pass
        self._provisional_vk_device = None
        self._provisional_vk_instance = None

        if xr is not None and self.instance is not None:
            try:
                xr.destroy_instance(self.instance)
            except Exception:
                pass
            self.instance = None

        self.system_id = None
        self.swapchain_format = None
        self._graphics_binding = None
        self._initialized = False
        self._drop_output_frames()
        self._has_presented_frame = False
        self._last_quad_layers = []
        self._source_frame_wait_logged = False
        self._accept_output = False
        self._filament_animation_origin = None
        self._profile_initial_head = None
        self._profile_space_applied = False
        self._reference_space_type = None

    def __enter__(self) -> "OpenXrVulkanPresenter":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _create_vulkan_objects(self, api_version: int) -> None:
        xr = self.xr
        import vulkan as vk

        self._vulkan_loader, self._vk_get_instance_proc_addr = _load_vulkan_proc_addr(xr)
        platform = _openxr_platform_module(xr)

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=self.config.application_name,
            applicationVersion=1,
            pEngineName="D2S",
            engineVersion=1,
            apiVersion=int(api_version),
        )
        instance_create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        xr_instance, vulkan_result = xr.create_vulkan_instance_khr(
            self.instance,
            xr.VulkanInstanceCreateInfoKHR(
                system_id=self.system_id,
                pfn_get_instance_proc_addr=self._vk_get_instance_proc_addr,
                vulkan_create_info=_cffi_struct_pointer(
                    vk, instance_create_info, platform.VkInstanceCreateInfo
                ),
            ),
        )
        _check_vulkan_result(vulkan_result, "xrCreateVulkanInstanceKHR")
        vk_instance = _ctypes_handle_to_cffi(vk, "VkInstance", xr_instance)
        self._provisional_vk_instance = vk_instance

        xr_physical_device = xr.get_vulkan_graphics_device2_khr(
            self.instance,
            xr.VulkanGraphicsDeviceGetInfoKHR(
                system_id=self.system_id,
                vulkan_instance=xr_instance,
            ),
        )
        vk_physical_device = _ctypes_handle_to_cffi(
            vk, "VkPhysicalDevice", xr_physical_device
        )
        queue_family_index = find_graphics_queue_family(vk, vk_physical_device)
        try:
            timeline_features, synchronization2_enabled = _require_timeline_semaphore_features(
                vk, vk_physical_device
            )
        except VulkanCapabilityError as exc:
            raise OpenXrVulkanUnavailableError(str(exc)) from exc
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        # XR_KHR_vulkan_enable2 does not expose xrGetVulkanDeviceExtensionsKHR.
        # Device extensions are selected from the application's Vulkan resource
        # requirements and validated against the runtime-selected physical device.
        external_extensions = VulkanExportableImage.required_device_extensions()
        available_extensions = {
            _decode_name(item.extensionName)
            for item in vk.vkEnumerateDeviceExtensionProperties(vk_physical_device, None)
        }
        missing_extensions = [
            name for name in external_extensions if name not in available_extensions
        ]
        if missing_extensions:
            raise OpenXrVulkanUnavailableError(
                "Vulkan external-memory extensions are unavailable: "
                + ", ".join(missing_extensions)
            )
        optional_external_semaphore = (
            VulkanExportableImage.optional_external_semaphore_extensions()
        )
        enabled_optional = (
            optional_external_semaphore
            if optional_external_semaphore
            and all(name in available_extensions for name in optional_external_semaphore)
            else ()
        )
        device_extensions = tuple(
            dict.fromkeys((*external_extensions, *enabled_optional))
        )
        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=timeline_features,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(device_extensions),
            ppEnabledExtensionNames=list(device_extensions),
        )
        xr_device, vulkan_result = xr.create_vulkan_device_khr(
            self.instance,
            xr.VulkanDeviceCreateInfoKHR(
                system_id=self.system_id,
                pfn_get_instance_proc_addr=self._vk_get_instance_proc_addr,
                vulkan_physical_device=xr_physical_device,
                vulkan_create_info=_cffi_struct_pointer(
                    vk, device_create_info, platform.VkDeviceCreateInfo
                ),
            ),
        )
        _check_vulkan_result(vulkan_result, "xrCreateVulkanDeviceKHR")
        vk_device = _ctypes_handle_to_cffi(vk, "VkDevice", xr_device)
        self._provisional_vk_device = vk_device
        self.vulkan = VulkanContext.adopt(
            instance=vk_instance,
            physical_device=vk_physical_device,
            device=vk_device,
            queue_family_index=queue_family_index,
            owns_instance=True,
            owns_device=True,
            timeline_semaphore_enabled=True,
            synchronization2_enabled=synchronization2_enabled,
        )
        self._provisional_vk_device = None
        self._provisional_vk_instance = None
        self._graphics_binding = xr.GraphicsBindingVulkan2KHR(
            instance=xr_instance,
            physical_device=xr_physical_device,
            device=xr_device,
            queue_family_index=queue_family_index,
            queue_index=0,
        )

    def _create_session_and_swapchains(self) -> None:
        xr = self.xr
        vk = self.vulkan.vk
        self._view_configuration_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self._environment_blend_mode = xr.EnvironmentBlendMode.OPAQUE
        self.session = xr.create_session(
            self.instance,
            xr.SessionCreateInfo(
                system_id=self.system_id,
                next=ctypes.cast(
                    ctypes.pointer(self._graphics_binding), ctypes.c_void_p
                ),
            ),
        )
        available_spaces = xr.enumerate_reference_spaces(self.session)
        self._reference_space_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self.reference_space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=self._reference_space_type
            ),
        )
        print(
            f"[OpenXRViewer] Reference space selected: "
            f"{getattr(self._reference_space_type, 'name', self._reference_space_type)}",
            flush=True,
        )
        formats = list(xr.enumerate_swapchain_formats(self.session))
        self.swapchain_format = _select_swapchain_format(
            vk, formats, self.config.swapchain_color_mode
        )
        print(
            "OpenXR swapchain color mode: "
            f"requested={self.config.swapchain_color_mode} "
            f"selected={_vulkan_format_name(vk, self.swapchain_format)} "
            f"format={self.swapchain_format}",
            flush=True,
        )
        view_configs = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self._view_configuration_type
        )
        if len(view_configs) < 2:
            raise OpenXrVulkanUnavailableError(
                f"PRIMARY_STEREO returned {len(view_configs)} view(s)"
            )

        for view_config in view_configs[:2]:
            width = _scaled_dimension(
                view_config.recommended_image_rect_width,
                view_config.max_image_rect_width,
                self.config.render_scale,
            )
            height = _scaled_dimension(
                view_config.recommended_image_rect_height,
                view_config.max_image_rect_height,
                self.config.render_scale,
            )
            handle = xr.create_swapchain(
                self.session,
                xr.SwapchainCreateInfo(
                    usage_flags=(
                        xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                        | xr.SwapchainUsageFlags.TRANSFER_DST_BIT
                    ),
                    format=self.swapchain_format,
                    sample_count=1,
                    width=width,
                    height=height,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                ),
            )
            images = list(
                xr.enumerate_swapchain_images(handle, xr.SwapchainImageVulkan2KHR)
            )
            if not images:
                xr.destroy_swapchain(handle)
                raise OpenXrVulkanUnavailableError(
                    "OpenXR runtime returned an empty Vulkan swapchain"
                )
            self.swapchains.append(
                _EyeSwapchain(
                    handle=handle,
                    images=images,
                    width=width,
                    height=height,
                    resources=self._register_swapchain_images(images, width, height),
                )
            )

    def _register_swapchain_images(
        self, images: list[Any], width: int, height: int,
        format_value: int | None = None,
    ) -> list[VulkanImageResource]:
        resources: list[VulkanImageResource] = []
        try:
            for index, item in enumerate(images):
                image = self.vulkan.image_handle_from_address(
                    _ctypes_handle_address(item.image)
                )
                resource = VulkanImageResource(
                    context=self.vulkan,
                    image=image,
                    view=None,
                    width=width,
                    height=height,
                    format=int(format_value if format_value is not None else self.swapchain_format),
                    layout=self.vulkan.vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    access_mask=0,
                    stage_mask=0,
                    queue_family_index=self.vulkan.queue_family_index,
                    external=True,
                    label=f"openxr-swapchain-{index}",
                )
                self.vulkan.register_external_image(resource)
                resources.append(resource)
        except Exception:
            for resource in reversed(resources):
                try:
                    self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            raise
        return resources

    def submit_output(self, frame: VulkanStereoOutputFrame) -> None:
        """Queue the newest Vulkan left/right frame for the next XR frame."""

        if not self._accept_output or not self.session_running:
            raise RuntimeError("OpenXR presenter is waiting for headset rendering")

        if not isinstance(frame.left_eye, VulkanImageResource) or not isinstance(
            frame.right_eye, VulkanImageResource
        ):
            raise TypeError("OpenXR Vulkan output requires VulkanImageResource eyes")
        if frame.left_eye.context is not self.vulkan or frame.right_eye.context is not self.vulkan:
            raise ValueError("OpenXR output images belong to a different Vulkan context")
        with self._output_lock:
            previous = self._pending_output
            self._pending_output = frame
        if previous is not None and previous is not frame:
            self._release_output_frame(previous)

    @staticmethod
    def _release_output_frame(frame: VulkanStereoOutputFrame | None) -> None:
        if frame is None:
            return
        callback = (frame.metadata or {}).get("_vulkan_output_release")
        if callable(callback):
            callback(frame.frame_id)

    def _drop_output_frames(self) -> None:
        with self._output_lock:
            pending = self._pending_output
            displayed = self._displayed_output
            rendering = self._rendering_output
            self._pending_output = None
            self._displayed_output = None
            self._rendering_output = None
        self._release_output_frame(pending)
        if displayed is not pending and displayed is not rendering:
            self._release_output_frame(displayed)
        if rendering is not pending and rendering is not displayed:
            self._release_output_frame(rendering)

    def _commit_output_frame(self, frame: VulkanStereoOutputFrame) -> None:
        with self._output_lock:
            previous = self._displayed_output
            if self._pending_output is frame:
                self._pending_output = None
            if self._rendering_output is frame:
                self._rendering_output = None
            self._displayed_output = frame
        if previous is not None and previous is not frame:
            self._release_output_frame(previous)

    def _abort_output_frame(self, frame: VulkanStereoOutputFrame) -> None:
        with self._output_lock:
            if self._rendering_output is frame:
                self._rendering_output = None
            if self._pending_output is frame:
                self._pending_output = None
        self._release_output_frame(frame)

    def _initialize_filament_bridges(self) -> None:
        bridge_path = self.config.filament_bridge_path or os.environ.get(
            "D2S_FILAMENT_BRIDGE"
        )
        if not bridge_path:
            return

        from .filament_vulkan_bridge import FilamentVulkanBridge

        bridge = FilamentVulkanBridge(bridge_path)
        try:
            bridge.create(
                instance=self.vulkan.instance,
                physical_device=self.vulkan.physical_device,
                device=self.vulkan.device,
                queue_family_index=self.vulkan.queue_family_index,
                queue_index=0,
            )
            for eye_index, eye in enumerate(self.swapchains):
                bridge.create_eye_swapchain(
                    eye_index,
                    (image.image for image in eye.images),
                    format=self.swapchain_format,
                    width=eye.width,
                    height=eye.height,
                )
            glb_path = self.config.filament_glb_path
            if glb_path:
                bridge.load_glb(Path(glb_path).read_bytes())
            if (
                self._controller_brand is not None
                and getattr(bridge, "controller_abi_available", True)
                and hasattr(bridge, "load_controller")
            ):
                bridge.load_controller(0, self._controller_brand.left_glb.read_bytes())
                bridge.load_controller(1, self._controller_brand.right_glb.read_bytes())
                print(
                    "Filament controllers loaded: "
                    f"brand={self._controller_brand.name} "
                    f"abi={bridge.controller_abi_available} "
                    f"visibility_abi={getattr(bridge, 'controller_visibility_abi_available', False)} "
                    f"laser_abi={getattr(bridge, 'laser_abi_available', False)}",
                    flush=True,
                )
            if self._filament_screen is not None:
                position, width, height, rotation = self._filament_screen
                bridge.create_screen()
                bridge.set_screen(position, width, height, rotation)
                print(
                    "Filament screen loaded: "
                    f"position={position} size={width:.3f}x{height:.3f} "
                    f"rotation={rotation}",
                    flush=True,
                )
                print(
                    "Filament screen image path: "
                    f"enabled={self._filament_screen_image_enabled} "
                    f"abi={getattr(bridge, 'screen_image_abi_available', False)} "
                    "fallback=OpenXR Quad Layer Vulkan copy",
                    flush=True,
                )
            bridge.set_scene_exposure(self._filament_scene_exposure)
            bridge.set_skybox_brightness(self._filament_skybox_brightness)
            bridge.set_fill_light(
                self._filament_fill_light_color,
                self._filament_fill_light_intensity,
                self._filament_fill_light_direction,
            )
            self.filament_bridge = bridge
        except Exception:
            bridge.close()
            self.filament_bridge = None
            raise

    def _update_filament_controllers(self, bridge: Any) -> None:
        if (
            self._controller_brand is None
            or not getattr(bridge, "controller_abi_available", True)
            or not hasattr(bridge, "set_controller_pose")
            or not hasattr(bridge, "set_controller_inputs")
        ):
            return
        offset = np.eye(4, dtype=np.float32)
        offset[:3, 3] = np.asarray(self._controller_brand.offset, dtype=np.float32)
        # Controller profiles use the legacy model calibration convention:
        # model_rotation_deg is a rotation around the local X axis.
        rotation = euler_to_mat4(
            0.0, math.radians(self._controller_brand.rotation_deg), 0.0
        ).astype(np.float32)
        for hand, (grip_matrix, aim_matrix) in enumerate(
            zip((self._grip_mat_l, self._grip_mat_r), (self._aim_mat_l, self._aim_mat_r))
        ):
            last_move = self._laser_last_move_l if hand == 0 else self._laser_last_move_r
            active = (
                grip_matrix is not None
                and self._frame_now - float(last_move) <= self._LASER_HIDE_AFTER
            )
            if getattr(bridge, "controller_visibility_abi_available", False):
                bridge.set_controller_visible(hand, active)
            if not active:
                self._reset_smoothed_ray(hand)
                if getattr(bridge, "laser_abi_available", False):
                    bridge.set_controller_laser(
                        hand, np.eye(4, dtype=np.float32), visible=False
                    )
                continue
            model_matrix = grip_matrix @ rotation @ offset
            bridge.set_controller_pose(hand, model_matrix)
            values = self._controller_input(hand)
            button_mask = 0
            for bit, name in enumerate(
                ("a_button", "b_button", "x_button", "y_button", "menu_button")
            ):
                if values.get(name, 0.0) > 0.5:
                    button_mask |= 1 << bit
            if values.get("stick_click", 0.0) > 0.5:
                button_mask |= 1 << 5
            bridge.set_controller_inputs(
                hand,
                trigger=values.get("trigger", 0.0),
                grip=values.get("grip", 0.0),
                joystick_x=values.get("joystick_x", 0.0),
                joystick_y=values.get("joystick_y", 0.0),
                button_mask=button_mask,
            )
            if getattr(bridge, "laser_abi_available", False) and hasattr(bridge, "set_controller_laser"):
                if aim_matrix is None:
                    bridge.set_controller_laser(
                        hand, np.eye(4, dtype=np.float32), visible=False
                    )
                else:
                    smoothed_origin, direction = self._get_smoothed_ray(hand)
                    if smoothed_origin is None or direction is None:
                        smoothed_origin = (
                            grip_matrix[:3, 3] + grip_matrix[:3, 1] * 0.020
                        ).astype(np.float64)
                        direction = (-aim_matrix[:3, 2]).astype(np.float64)
                    direction /= max(float(np.linalg.norm(direction)), 1e-8)
                    right_axis = aim_matrix[:3, 0].astype(np.float64)
                    right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
                    # Match the legacy controller ray calibration: rotate the
                    # Aim -Z vector by 12 degrees around local X and start the
                    # beam just beyond the grip shell.
                    angle = math.radians(12.0)
                    direction = (
                        direction * math.cos(angle)
                        + np.cross(right_axis, direction) * math.sin(angle)
                        + right_axis
                        * float(np.dot(right_axis, direction))
                        * (1.0 - math.cos(angle))
                    )
                    direction /= max(float(np.linalg.norm(direction)), 1e-8)
                    beam_origin = (
                        smoothed_origin.astype(np.float64) + direction * 0.11
                    )
                    normal_axis = np.cross(right_axis, direction)
                    normal_axis /= max(float(np.linalg.norm(normal_axis)), 1e-8)
                    right_axis = np.cross(direction, normal_axis)
                    right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
                    laser_matrix = np.eye(4, dtype=np.float32)
                    laser_matrix[:3, 0] = (right_axis * 0.006).astype(np.float32)
                    laser_matrix[:3, 1] = (direction * 0.4).astype(np.float32)
                    laser_matrix[:3, 2] = (normal_axis * 0.006).astype(np.float32)
                    laser_matrix[:3, 3] = beam_origin.astype(np.float32)
                    bridge.set_controller_laser(hand, laser_matrix, visible=True)

    def _load_filament_profile(self) -> None:
        profile_path = self.config.filament_profile_path
        if not profile_path:
            return
        with open(profile_path, "r", encoding="utf-8-sig") as handle:
            profile = json.load(handle)
        if not isinstance(profile, dict):
            raise ValueError("Filament profile root must be an object")

        view_pose = profile.get("view_pose", profile.get("camera"))
        view_poses = profile.get("view_poses")
        if isinstance(view_poses, list) and view_poses:
            index = int(profile.get("view_pose_index", 0)) % len(view_poses)
            view_pose = view_poses[index]
        if not isinstance(view_pose, dict):
            raise ValueError("Filament profile does not contain a view pose")

        try:
            model_position = profile.get(
                "model_position", profile.get("position", [0.0, 0.0, 0.0])
            )
            if not isinstance(model_position, (list, tuple)) or len(model_position) < 3:
                model_position = [0.0, 0.0, 0.0]
            model_rotation_deg = profile.get("model_rotation_deg", [0.0, 0.0, 0.0])
            if not isinstance(model_rotation_deg, (list, tuple)) or len(model_rotation_deg) < 3:
                model_rotation_deg = [0.0, 0.0, 0.0]
            model_scale = profile.get("model_scale", [1.0, 1.0, 1.0])
            if not isinstance(model_scale, (list, tuple)) or len(model_scale) < 3:
                model_scale = [1.0, 1.0, 1.0]

            world_position_vec = np.asarray(
                [float(view_pose[key]) for key in ("x", "y", "z")],
                dtype=np.float32,
            )
            rotation_deg = view_pose.get("rotation_deg")
            if not isinstance(rotation_deg, (list, tuple)) or len(rotation_deg) < 3:
                rotation_deg = [float(view_pose.get("angle", 0.0)), 0.0, 0.0]
            rotation_rad = [math.radians(float(value)) for value in rotation_deg[:3]]

            # view_poses are authored in environment world coordinates while
            # the imported GLB and calibrated OpenXR space use GLB-local
            # coordinates. Match the legacy viewer by applying the inverse
            # model transform before rebasing the reference space.
            model_matrix = euler_to_mat4(
                *(math.radians(float(value)) for value in model_rotation_deg[:3])
            ).astype(np.float32)
            model_matrix[:3, 3] = np.asarray(model_position[:3], dtype=np.float32)
            scale = np.asarray(model_scale[:3], dtype=np.float32)
            model_matrix[:3, :3] = model_matrix[:3, :3] @ np.diag(scale)
            glb_position = (
                np.linalg.inv(model_matrix)
                @ np.append(world_position_vec, 1.0)
            )[:3]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Filament profile view pose contains invalid values") from exc

        transform = euler_to_mat4(*rotation_rad).astype(np.float32)
        transform[:3, 3] = np.asarray(glb_position, dtype=np.float32)
        self._profile_head_transform = transform
        self._profile_view_name = str(view_pose.get("name", "profile"))
        self._profile_near_plane = max(0.001, float(profile.get("xr_projection_near", 0.05)))
        self._profile_far_plane = max(
            self._profile_near_plane + 1.0,
            float(profile.get("xr_projection_far", 1000.0)),
        )
        self._filament_scene_exposure = float(
            profile.get("preview_exposure", self._filament_scene_exposure)
        )
        self._filament_skybox_brightness = float(
            profile.get("preview_skybox_brightness", self._filament_skybox_brightness)
        )
        fill_color = profile.get(
            "preview_fill_light_color", self._filament_fill_light_color
        )
        fill_direction = profile.get(
            "preview_fill_light_direction", self._filament_fill_light_direction
        )
        if isinstance(fill_color, (list, tuple)) and len(fill_color) >= 3:
            self._filament_fill_light_color = tuple(
                float(value) for value in fill_color[:3]
            )
        if isinstance(fill_direction, (list, tuple)) and len(fill_direction) >= 3:
            self._filament_fill_light_direction = tuple(
                float(value) for value in fill_direction[:3]
            )
        self._filament_fill_light_intensity = float(
            profile.get(
                "preview_fill_light_intensity", self._filament_fill_light_intensity
            )
        )
        screen = profile.get("screen")
        if isinstance(screen, dict):
            screen_position = screen.get("position", [0.0, 1.2, -2.0])
            rotation = screen.get("rotation_deg", [0.0, 0.0, 0.0])
            if (
                isinstance(screen_position, (list, tuple))
                and len(screen_position) >= 3
                and isinstance(rotation, (list, tuple))
                and len(rotation) >= 3
            ):
                self._filament_screen = (
                    tuple(float(value) for value in screen_position[:3]),
                    float(screen.get("width", 2.4)),
                    float(screen.get(
                        "height",
                        float(screen.get("width", 2.4)) * 9.0 / 16.0,
                    )),
                    tuple(float(value) for value in rotation[:3]),
                )
                self._filament_screen_initial = self._filament_screen
        print(
            f"Loaded Filament profile view: {self._profile_view_name} "
            f"world_position={world_position_vec.tolist()} glb_position={glb_position.tolist()} "
            f"rotation_rad={rotation_rad}",
            flush=True,
        )

    def _apply_filament_profile(self, views: list[Any]) -> list[Any]:
        # The environment profile is applied once by rebasing the shared
        # OpenXR reference space. Runtime eye views must remain unmodified so
        # the compositor receives the matching headset poses.
        return views

    def _apply_profile_reference_space(self, views: list[Any]) -> bool:
        """Apply the saved seat pose once, keeping subsequent views world-locked."""
        if self._profile_space_applied or self._profile_head_transform is None:
            return False
        if len(views) < 2 or self.xr is None or self.session is None:
            return False
        eye_matrices = [_xr_view_pose_to_model_mat4(view.pose) for view in views[:2]]
        raw_head = eye_matrices[0].copy()
        raw_head[:3, 3] = (eye_matrices[0][:3, 3] + eye_matrices[1][:3, 3]) * 0.5
        # Match the legacy environment path: keep the room level by removing
        # headset pitch/roll from the initial pose, then place the saved
        # profile pose in that stable world space.
        reference_head = self._level_head_model_mat4(raw_head)
        space_pose = reference_head @ np.linalg.inv(self._profile_head_transform)
        try:
            new_space = self.xr.create_reference_space(
                self.session,
                self.xr.ReferenceSpaceCreateInfo(
                    reference_space_type=(
                        self._reference_space_type
                        or self.xr.ReferenceSpaceType.LOCAL
                    ),
                    pose_in_reference_space=mat4_to_xr_posef(space_pose.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to apply profile reference space: {exc}", flush=True)
            return False
        old_space = self.reference_space
        self.reference_space = new_space
        # Controller action spaces must use the same calibrated world space.
        self._xr_space = new_space
        self._profile_space_applied = True
        self._profile_initial_head = raw_head
        if old_space is not None:
            try:
                self.xr.destroy_space(old_space)
            except Exception:
                pass
        print("[OpenXRViewer] Applied profile pose to stable OpenXR reference space", flush=True)
        return True

    @staticmethod
    def _level_head_model_mat4(head_mat: np.ndarray) -> np.ndarray:
        """Keep position and yaw while preserving a level environment."""
        pos = head_mat[:3, 3].copy()
        forward = -head_mat[:3, 2].astype(np.float32)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        yaw = 0.0 if norm < 1e-6 else math.atan2(
            -float(forward[0] / norm), -float(forward[2] / norm)
        )
        leveled = euler_to_mat4(yaw, 0.0, 0.0).astype(np.float32)
        leveled[:3, 3] = pos
        return leveled

    def _render_projection_layer(
        self,
        views: list[Any],
        output_frame: VulkanStereoOutputFrame | None | object = _OUTPUT_FRAME_UNSET,
    ) -> Any | None:
        if len(views) < len(self.swapchains):
            return None
        # The profile adjusts the Filament camera relative to the model. The
        # composition layer must retain the runtime-provided eye poses so the
        # OpenXR compositor keeps the rendered image aligned with the headset.
        composition_views = views
        render_views = self._apply_filament_profile(views)
        xr = self.xr
        if output_frame is _OUTPUT_FRAME_UNSET:
            with self._output_lock:
                output_frame = self._pending_output
                self._pending_output = None
        else:
            with self._output_lock:
                if self._pending_output is output_frame:
                    self._pending_output = None
        if isinstance(output_frame, VulkanStereoOutputFrame):
            with self._output_lock:
                self._rendering_output = output_frame
        if self._filament_animation_origin is None:
            self._filament_animation_origin = self._frame_now
        animation_time = max(0.0, self._frame_now - self._filament_animation_origin)
        acquired_images: list[tuple[_EyeSwapchain, int]] = []
        render_succeeded = False
        try:
            # Keep both OpenXR images acquired while Filament queues both eye
            # submissions. They are released only after the single frame-wide
            # completion wait below.
            for eye in self.swapchains:
                image_index = xr.acquire_swapchain_image(eye.handle)
                acquired_images.append((eye, image_index))
                xr.wait_swapchain_image(
                    eye.handle,
                    xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
            screen_image_projection = self._can_use_filament_screen_image(output_frame)
            for eye_index, (eye, image_index) in enumerate(acquired_images):
                if self.filament_bridge is not None:
                    bridge = self.filament_bridge
                    bridge.set_active_eye(eye_index)
                    # Do not pass a raw Vulkan VkImage to Filament's generic
                    # Texture::Builder::import API. The old validated path
                    # presents the virtual screen as an OpenXR quad layer.
                    _update_filament_camera(
                        bridge,
                        render_views[eye_index],
                        near_plane=self._profile_near_plane,
                        far_plane=self._profile_far_plane,
                    )
                    if (
                        output_frame is not None
                        and screen_image_projection
                        and self._filament_screen is not None
                    ):
                        screen_source = (
                            output_frame.left_eye
                            if eye_index == 0
                            else output_frame.right_eye
                        )
                        # Bind the matching runtime eye image to the Filament
                        # screen material. The native bridge caches imported
                        # VkImages by handle, so this is not a per-frame
                        # texture allocation.
                        bridge.set_screen_image(
                            screen_source.image,
                            width=screen_source.width,
                            height=screen_source.height,
                            format=screen_source.format,
                        )
                        ready_key = (
                            "vulkan_ready_semaphore_left"
                            if eye_index == 0
                            else "vulkan_ready_semaphore_right"
                        )
                        ready_semaphore = (output_frame.metadata or {}).get(ready_key)
                        if (
                            ready_semaphore is not None
                            and getattr(
                                bridge, "screen_ready_semaphore_abi_available", False
                            )
                        ):
                            bridge.set_screen_ready_semaphore(ready_semaphore)
                    self._update_filament_controllers(bridge)
                    if hasattr(bridge, "apply_animations"):
                        bridge.apply_animations(animation_time)
                    bridge.set_acquired_image(image_index)
                    bridge.begin_frame()
                    bridge.end_frame()
                else:
                    if output_frame is not None:
                        source = (
                            output_frame.left_eye
                            if eye_index == 0
                            else output_frame.right_eye
                        )
                        self.vulkan.copy_image(
                            source,
                            eye.resources[image_index],
                            wait_for_timeline=output_frame.ready_timeline,
                        )
                    else:
                        image_address = _ctypes_handle_address(eye.images[image_index].image)
                        image = self.vulkan.image_handle_from_address(image_address)
                        self.vulkan.clear_color_image(image, self.config.clear_color)
            if self.filament_bridge is not None:
                bridge = self.filament_bridge
                if getattr(bridge, "async_submit_abi_available", False):
                    # Both eyes are submitted before the single completion wait.
                    bridge.wait_for_idle()
            render_succeeded = True
        finally:
            for eye, _image_index in acquired_images:
                xr.release_swapchain_image(eye.handle)
            if (
                isinstance(output_frame, VulkanStereoOutputFrame)
                and not render_succeeded
            ):
                self._abort_output_frame(output_frame)
        return OpenXrCompositionBuilder(xr, self.reference_space).projection_layer(
            composition_views, self.swapchains
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("OpenXrVulkanPresenter is not initialized")

    def _ensure_quad_swapchains(self, width: int, height: int) -> None:
        if self._quad_swapchain_extent == (width, height) and len(self._quad_swapchains) == 2:
            return
        if self.xr is None or self.session is None or self.vulkan is None:
            return
        self._destroy_quad_swapchains()
        vk = self.vulkan.vk
        formats = list(self.xr.enumerate_swapchain_formats(self.session))
        # The runtime output contract is display-referred sRGB. Match the
        # validated legacy Quad Layer path and prefer an sRGB target.
        quad_format = _select_swapchain_format(vk, formats, "srgb")
        for _ in range(2):
            handle = self.xr.create_swapchain(
                self.session,
                self.xr.SwapchainCreateInfo(
                    usage_flags=(self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                                 | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT),
                    format=quad_format, sample_count=1, width=width, height=height,
                    face_count=1, array_size=1, mip_count=1,
                ),
            )
            images = list(self.xr.enumerate_swapchain_images(
                handle, self.xr.SwapchainImageVulkan2KHR
            ))
            self._quad_swapchains.append(_EyeSwapchain(
                handle, images, width, height,
                self._register_swapchain_images(images, width, height, quad_format),
            ))
        self._quad_swapchain_format = int(quad_format)
        self._quad_swapchain_extent = (width, height)
        print(
            f"[OpenXRViewer] Quad layer swapchains created: "
            f"format={_vulkan_format_name(vk, quad_format)} extent={width}x{height}",
            flush=True,
        )

    def _destroy_quad_swapchains(self) -> None:
        if self.xr is None:
            self._quad_swapchains.clear()
            return
        for eye in reversed(self._quad_swapchains):
            for resource in reversed(eye.resources):
                try:
                    if self.vulkan is not None:
                        self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            try:
                self.xr.destroy_swapchain(eye.handle)
            except Exception:
                pass
        self._quad_swapchains.clear()
        self._quad_swapchain_format = None
        self._quad_swapchain_extent = None

    def _destroy_tool_quad_layers(self) -> None:
        for entry in self._overlay_quad_entries.values():
            try:
                entry["staging"].close()
            except Exception:
                pass
            for resource in reversed(entry.get("resources", ())):
                try:
                    if self.vulkan is not None:
                        self.vulkan.unregister_external_image(resource)
                except Exception:
                    pass
            try:
                if self.xr is not None:
                    self.xr.destroy_swapchain(entry["swapchain"])
            except Exception:
                pass
        self._overlay_quad_entries.clear()

    def _render_quad_layers(self, output_frame: VulkanStereoOutputFrame | None) -> list[Any]:
        # The main virtual screen is rendered in the Projection Layer when
        # the per-eye Filament image path is enabled. Keep this function for
        # controller tools and other 2D overlays only.
        if output_frame is None:
            return []
        if self._filament_screen is None:
            return []
        layers = self._render_tool_quad_layers()
        screen_in_projection = self._can_use_filament_screen_image(output_frame)
        if screen_in_projection:
            return layers
        width = int(output_frame.left_eye.width)
        height = int(output_frame.left_eye.height)
        self._ensure_quad_swapchains(width, height)
        if len(self._quad_swapchains) < 2:
            return []
        position, screen_width, screen_height, rotation = self._filament_screen
        layers = list(layers)
        for eye_index, eye in enumerate(self._quad_swapchains):
            source = output_frame.left_eye if eye_index == 0 else output_frame.right_eye
            with _acquired_swapchain_image(self.xr, eye) as image_index:
                # The output contract is top-left and the Vulkan swapchain
                # image uses the same row order. Do not apply a second Y
                # transform here; screen pose is handled independently below.
                self.vulkan.copy_image(
                    source,
                    eye.resources[image_index],
                    flip_y=False,
                )
            layers.append(OpenXrCompositionBuilder(
                self.xr, self.reference_space
            ).quad_layer(
                eye, position, screen_width, screen_height, rotation, eye_index
            ))
        return layers

    def _render_tool_quad_layers(self) -> list[Any]:
        """Submit legacy keyboard, laser, FPS, aperture and help quads."""
        if self.xr is None or self.session is None or self.vulkan is None:
            return []
        position, width, height, rotation = self._filament_screen or ((0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0))
        specs = []
        if self._keyboard_visible:
            keyboard_width = float(self._keyboard_width)
            keyboard_height = float(self._keyboard_height)
            rgba, self._keyboard_keys = build_keyboard_rgba(
                self._kb_show_shifted, keyboard_width, keyboard_height
            )
            keyboard_pose = self._keyboard_pose_mat4()
            specs.append((
                "keyboard", rgba,
                tuple(float(value) for value in keyboard_pose[:3, 3]),
                (keyboard_width, keyboard_height), rotation,
            ))
        if self._fps_overlay_visible:
            rgba = build_fps_overlay_rgba(
                actual_fps=0.0, sbs_fps=0.0, latency_ms=0.0,
                screen_width=width, screen_height=height, screen_distance=abs(float(position[2])),
                depth_strength=0.0, vr_res=(0, 0), sbs_res=(0, 0),
                controller_brand=getattr(self._controller_brand, "name", ""),
                environment_visible=True,
            )
            specs.append(("fps", rgba, (position[0] - width * 0.42, position[1] + height * 0.72, position[2]), (width * 0.42, height * 0.13), rotation))
        if self._operation_guide_visible:
            rgba = build_help_rgba(environment_mode=False)
            specs.append(("help", rgba, (position[0] + width * 0.34, position[1] + height * 0.72, position[2]), (width * 0.32, height * 0.28), rotation))
        if self._aperture_visible:
            rgba = build_short_osd_rgba(("Aperture", "B: close"), width=384, height=64)
            specs.append(("aperture", rgba, position, (width * 0.24, height * 0.06), rotation))
        return [self._upload_tool_quad(*spec) for spec in specs]

    def _upload_tool_quad(self, key, rgba, position, size, rotation):
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        entry = self._overlay_quad_entries.get(key)
        formats = self.xr.enumerate_swapchain_formats(self.session)
        format_value = _select_swapchain_format(self.vulkan.vk, list(formats), "srgb")
        if entry is None or entry["size"] != (width, height):
            if entry is not None:
                entry["staging"].close()
                for resource in reversed(entry["resources"]):
                    self.vulkan.unregister_external_image(resource)
                self.xr.destroy_swapchain(entry["swapchain"])
            swapchain = self.xr.create_swapchain(
                self.session,
                self.xr.SwapchainCreateInfo(
                    usage_flags=(self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT),
                    format=format_value, sample_count=1, width=width, height=height,
                    face_count=1, array_size=1, mip_count=1,
                ),
            )
            images = list(self.xr.enumerate_swapchain_images(swapchain, self.xr.SwapchainImageVulkan2KHR))
            entry = {
                "swapchain": swapchain,
                "size": (width, height),
                "resources": self._register_swapchain_images(images, width, height, format_value),
                "staging": VulkanHostImage(self.vulkan, width, height, format=format_value, label=f"overlay-{key}"),
            }
            self._overlay_quad_entries[key] = entry
        entry["staging"].upload(rgba)
        with _acquired_swapchain_image(self.xr, _EyeSwapchain(entry["swapchain"], [], width, height, entry["resources"])) as image_index:
            self.vulkan.copy_image(entry["staging"].resource, entry["resources"][image_index])
        if len(rotation) == 4:
            qx, qy, qz, qw = (float(value) for value in rotation)
        else:
            qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            space=self.reference_space,
            eye_visibility=self.xr.EyeVisibility.BOTH,
            sub_image=self.xr.SwapchainSubImage(
                swapchain=entry["swapchain"],
                image_rect=self.xr.Rect2Di(offset=self.xr.Offset2Di(x=0, y=0), extent=self.xr.Extent2Di(width=width, height=height)),
                image_array_index=0,
            ),
            pose=self.xr.Posef(
                orientation=self.xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=self.xr.Vector3f(x=float(position[0]), y=float(position[1]), z=float(position[2])),
            ),
            size=self.xr.Extent2Df(width=float(size[0]), height=float(size[1])),
        )


@contextmanager
def _acquired_swapchain_image(xr: Any, eye: _EyeSwapchain):
    """Guarantee release after every successful acquire, including wait errors."""

    image_index = xr.acquire_swapchain_image(eye.handle)
    try:
        xr.wait_swapchain_image(
            eye.handle,
            xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
        )
        yield image_index
    finally:
        xr.release_swapchain_image(eye.handle)


def _xr_view_pose_to_model_mat4(pose: Any) -> np.ndarray:
    matrix = _xr_quat_to_mat4(pose.orientation).astype(np.float32)
    matrix[:3, 3] = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    return matrix


def _euler_degrees_to_quaternion(rotation: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert legacy profile yaw/pitch/roll degrees to OpenXR xyzw."""
    yaw, pitch, roll = (
        math.radians(float(value)) for value in rotation[:3]
    )
    matrix = euler_to_mat4(yaw, pitch, roll)
    return tuple(float(value) for value in _mat3_to_quat_xyzw(matrix[:3, :3]))


def _update_filament_camera(
    bridge: Any,
    view: Any,
    *,
    near_plane: float = 0.05,
    far_plane: float = 1000.0,
) -> None:
    pose = view.pose
    rotation = _xr_quat_to_mat4(pose.orientation)[:3, :3]
    position = (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
    )
    forward = rotation @ (0.0, 0.0, -1.0)
    up = rotation @ (0.0, 1.0, 0.0)
    center = tuple(position[index] + float(forward[index]) for index in range(3))
    bridge.set_camera_look_at(position, center, tuple(float(value) for value in up))

    fov = view.fov
    left = math.tan(float(fov.angle_left)) * near_plane
    right = math.tan(float(fov.angle_right)) * near_plane
    bottom = math.tan(float(fov.angle_down)) * near_plane
    top = math.tan(float(fov.angle_up)) * near_plane
    if hasattr(bridge, "set_camera_projection_frustum"):
        bridge.set_camera_projection_frustum(
            left, right, bottom, top,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        return
    horizontal = max(0.01, abs(float(fov.angle_right) - float(fov.angle_left)))
    vertical = max(0.01, abs(float(fov.angle_up) - float(fov.angle_down)))
    aspect = math.tan(horizontal * 0.5) / max(math.tan(vertical * 0.5), 1e-6)
    bridge.set_camera_projection(
        math.degrees(vertical),
        aspect,
        near_plane=near_plane,
        far_plane=far_plane,
    )


def _import_openxr() -> Any:
    try:
        import xr
    except (ImportError, OSError) as exc:
        raise OpenXrVulkanUnavailableError(
            "pyopenxr or the OpenXR loader is unavailable"
        ) from exc
    return xr


def _get_vulkan_graphics_requirements2(
    xr: Any, instance: Any, system_id: Any
) -> Any:
    function = ctypes.cast(
        xr.get_instance_proc_addr(
            instance.instance, "xrGetVulkanGraphicsRequirements2KHR"
        ),
        xr.platform.PFN_xrGetVulkanGraphicsRequirements2KHR,
    )
    requirements = xr.GraphicsRequirementsVulkan2KHR()
    result = xr.check_result(function(instance, system_id, ctypes.byref(requirements)))
    if result.is_exception():
        raise result
    return requirements


def _select_vulkan_api_version(requirements: Any, requested: int) -> int:
    minimum = make_vulkan_version(
        requirements.min_api_version_supported.major,
        requirements.min_api_version_supported.minor,
        requirements.min_api_version_supported.patch,
    )
    maximum = make_vulkan_version(
        requirements.max_api_version_supported.major,
        requirements.max_api_version_supported.minor,
        requirements.max_api_version_supported.patch,
    )
    if minimum > maximum:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime returned an invalid Vulkan API version range"
        )
    if maximum < MIN_VULKAN_API_VERSION:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime does not support the required Vulkan 1.2 minimum"
        )
    selected = max(minimum, min(int(requested), maximum))
    if selected < MIN_VULKAN_API_VERSION:
        raise OpenXrVulkanUnavailableError(
            "Negotiated Vulkan API version is below the required Vulkan 1.2 minimum"
        )
    return selected


def _select_swapchain_format(
    vk: Any, available_formats: list[int], color_mode: str = "srgb"
) -> int:
    mode = str(color_mode or "srgb").strip().lower()
    if mode not in {"srgb", "unorm", "auto"}:
        raise ValueError("OpenXR swapchain color mode must be srgb, unorm, or auto")

    srgb = (
        vk.VK_FORMAT_R8G8B8A8_SRGB,
        vk.VK_FORMAT_B8G8R8A8_SRGB,
    )
    unorm = (
        vk.VK_FORMAT_R8G8B8A8_UNORM,
        vk.VK_FORMAT_B8G8R8A8_UNORM,
    )
    if mode == "unorm":
        preferred = unorm + srgb
    else:
        preferred = srgb + unorm
    for candidate in preferred:
        if int(candidate) in available_formats:
            return int(candidate)
    if not available_formats:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime returned no swapchain formats"
        )
    return int(available_formats[0])


def _vulkan_format_name(vk: Any, value: int) -> str:
    names = {
        int(vk.VK_FORMAT_R8G8B8A8_SRGB): "R8G8B8A8_SRGB",
        int(vk.VK_FORMAT_B8G8R8A8_SRGB): "B8G8R8A8_SRGB",
        int(vk.VK_FORMAT_R8G8B8A8_UNORM): "R8G8B8A8_UNORM",
        int(vk.VK_FORMAT_B8G8R8A8_UNORM): "B8G8R8A8_UNORM",
    }
    return names.get(int(value), "runtime-preferred")


def _scaled_dimension(recommended: int, maximum: int, scale: float) -> int:
    return max(1, min(int(maximum), round(int(recommended) * float(scale))))


def _openxr_platform_module(xr: Any) -> Any:
    return importlib.import_module(xr.VulkanInstanceCreateInfoKHR.__module__)


def _load_vulkan_proc_addr(xr: Any) -> tuple[Any, Any]:
    if sys.platform == "win32":
        candidates = ["vulkan-1.dll"]
    elif sys.platform == "darwin":
        candidates = ["libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib"]
    else:
        candidates = ["libvulkan.so.1", "libvulkan.so"]
    discovered = ctypes.util.find_library("vulkan")
    if discovered:
        candidates.append(discovered)

    platform = _openxr_platform_module(xr)
    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            loader = (
                ctypes.WinDLL(candidate)
                if sys.platform == "win32"
                else ctypes.CDLL(candidate)
            )
            function = ctypes.cast(
                loader.vkGetInstanceProcAddr, platform.PFN_vkGetInstanceProcAddr
            )
            return loader, function
        except (AttributeError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise OpenXrVulkanUnavailableError(
        "Unable to load vkGetInstanceProcAddr: " + "; ".join(errors)
    )


def _cffi_struct_pointer(vk: Any, value: Any, ctypes_type: Any) -> Any:
    address = int(vk.ffi.cast("uintptr_t", vk.ffi.addressof(value)))
    return ctypes.cast(ctypes.c_void_p(address), ctypes.POINTER(ctypes_type))


def _ctypes_handle_to_cffi(vk: Any, type_name: str, handle: Any) -> Any:
    address = _ctypes_handle_address(handle)
    if not address:
        raise OpenXrVulkanUnavailableError(f"OpenXR returned a null {type_name}")
    return vk.ffi.cast(type_name, address)


def _ctypes_handle_address(handle: Any) -> int:
    return int(ctypes.cast(handle, ctypes.c_void_p).value or 0)


def _check_vulkan_result(result: Any, operation: str) -> None:
    value = int(result.value if hasattr(result, "value") else result)
    if value != 0:
        raise OpenXrVulkanUnavailableError(f"{operation} returned VkResult {value}")


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value)
