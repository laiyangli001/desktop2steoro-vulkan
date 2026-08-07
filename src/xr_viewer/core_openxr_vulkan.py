from __future__ import annotations

import ctypes
import ctypes.util
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import importlib
import json
import math
import os
import queue
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
    ImageState,
    VulkanContext,
    VulkanCapabilityError,
    _require_timeline_semaphore_features,
    find_graphics_queue_family,
    make_vulkan_version,
)
from viewer.vulkan_resources import VulkanExportableImage, VulkanHostImage, VulkanImageResource
from viewer.vulkan_msdf_quad import VulkanMsdfQuadRenderer, VulkanMsdfQuadRequest
from viewer.vulkan_projection_screen import VulkanProjectionScreenPass
from app_runtime.output_contract import VulkanStereoOutputFrame


_OUTPUT_FRAME_UNSET = object()

from .core_controller_actions import CoreControllerActionsMixin
from .core_input_helpers import CoreInputHelpersMixin
from .core_controller_input import CoreControllerInputMixin
from .core_controller_guide_input import CoreControllerGuideInputMixin
from .core_controller_shortcuts import CoreControllerShortcutsMixin
from .core_controller_pose import CoreControllerPoseMixin
from .core_controller_ray import CoreControllerRayMixin
from .controller_models import (
    controller_button_local_position,
    discover_controller_brands,
    select_controller_brand,
)
from viewer.controller_help import get_controller_help_rows
from .filters import OneEuroFilter3D
from .xr_math import (
    _fov_to_proj_mat4_d3d,
    _mat3_to_quat_xyzw,
    _pose_to_view_mat4,
    _xr_quat_to_mat4,
    euler_to_mat4,
    mat4_to_xr_posef,
)
from .overlay_textures import (
    build_controller_callout_rgba,
    build_cursor_rgba,
    build_fps_overlay_rgba,
    build_help_rgba,
    build_team_help_rgba,
    build_keyboard_rgba,
    build_screen_adjust_osd_rgba,
    build_screen_preset_osd_rgba,
    build_short_osd_rgba,
)
from .keyboard_layout import _KB_TEX_H, _KB_TEX_W
from .msdf_font_atlas import MsdfFontAtlas
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
from utils import LANG
from utils.xr_headset_presets import resolve_xr_headset_preset
from utils.screen_resolution_policy import (
    ScreenSamplingPlan,
    build_screen_sampling_plan,
)


_DEFAULT_XR_HEADSET_PRESET = resolve_xr_headset_preset(None)

_MSDF_OSD_SCALE = 0.58
_MSDF_OSD_RUN_GAP = 8.0
_MSDF_OSD_PADDING_X = 20.0
_MSDF_OSD_PADDING_Y = 14.0
_MSDF_OSD_REFERENCE_HEIGHT = 78.0
_TOOL_OVERLAY_UPDATE_INTERVAL = 1.0

# Virtual Desktop does not accept the stereo screen Quad swapchain used by the
# reprojection experiment. Keep the implementation isolated for diagnosis, but
# never allow it to take ownership of the primary screen presentation path.
_SCREEN_QUAD_REPROJECTION_SUPPORTED = False


def _env_flag(name: str, default: bool = False) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _layout_msdf_osd_runs(
    atlas: MsdfFontAtlas,
    runs: tuple[tuple[str, tuple[int, int, int, int]], ...]
    | tuple[dict[str, Any], ...],
) -> tuple[int, int, tuple[dict[str, Any], ...]]:
    """Fit one-line MSDF runs to a padded Quad canvas."""
    normalized: list[tuple[str, tuple[int, int, int, int]]] = []
    for run in runs:
        if isinstance(run, dict):
            text = str(run.get("text", ""))
            color = tuple(run.get("color", (255, 255, 255, 255)))
        else:
            text, color = run
            text = str(text)
            color = tuple(color)
        if len(color) != 4:
            raise ValueError("MSDF OSD colors must contain four components")
        normalized.append((text, color))

    widths = [
        float(atlas.text_advance(text, scale=_MSDF_OSD_SCALE))
        for text, _color in normalized
    ]
    content_width = sum(widths) + _MSDF_OSD_RUN_GAP * max(0, len(widths) - 1)
    canvas_width = max(
        64,
        int(math.ceil(content_width + 2.0 * _MSDF_OSD_PADDING_X)),
    )
    canvas_height = max(
        48,
        int(
            math.ceil(
                float(atlas.line_height) * _MSDF_OSD_SCALE
                + 2.0 * _MSDF_OSD_PADDING_Y
            )
        ),
    )
    cursor = max(
        _MSDF_OSD_PADDING_X,
        (float(canvas_width) - content_width) * 0.5,
    )
    laid_out: list[dict[str, Any]] = []
    for (text, color), run_width in zip(normalized, widths):
        laid_out.append(
            {
                "text": text,
                "x": cursor,
                "y": _MSDF_OSD_PADDING_Y,
                "scale": _MSDF_OSD_SCALE,
                "color": color,
            }
        )
        cursor += run_width + _MSDF_OSD_RUN_GAP
    return canvas_width, canvas_height, tuple(laid_out)


def _build_msdf_panel_request(
    atlas: MsdfFontAtlas,
    runs: tuple[dict[str, Any], ...],
    *,
    width: int,
    height: int,
    background: tuple[int, int, int, int],
    radius: float = 14.0,
) -> VulkanMsdfQuadRequest:
    """Create a GPU MSDF panel request with explicit canvas geometry."""
    return VulkanMsdfQuadRequest(
        width=int(width),
        height=int(height),
        runs=tuple(runs),
        background=background,
        radius=float(radius),
    )


def _build_msdf_depth_osd_request(
    atlas: MsdfFontAtlas, depth_strength: float, message: str | None = None
) -> VulkanMsdfQuadRequest:
    """Build the legacy depth or stereo-mode prompt as a Quad MSDF panel."""
    if message:
        runs = ((str(message), (0, 210, 230, 255)),)
    else:
        runs = (
            ("Depth Strength", (150, 158, 185, 255)),
            (f"{max(0.0, float(depth_strength)):.2f}", (0, 210, 230, 255)),
        )
    width, height, laid_out = _layout_msdf_osd_runs(atlas, runs)
    return _build_msdf_panel_request(
        atlas,
        laid_out,
        width=width,
        height=height,
        background=(32, 32, 36, 210),
    )


def _build_msdf_fps_panel(
    atlas: MsdfFontAtlas,
    *,
    actual_fps: float,
    sbs_fps: float,
    latency_ms: float,
    screen_width: float,
    screen_height: float,
    screen_distance: float,
    depth_strength: float,
    vr_res: tuple[int, int],
    sbs_res: tuple[int, int],
    controller_brand: str,
    environment_visible: bool,
) -> VulkanMsdfQuadRequest:
    """Build the FPS panel as MSDF runs instead of a rasterized text bitmap."""
    scale = 24.0 / max(float(atlas.line_height), 1.0)
    label_color = (150, 158, 185, 255)
    value_colors = (
        (0, 230, 90, 255),
        (0, 210, 230, 255),
        (255, 190, 40, 255),
        (0, 210, 230, 255),
        (0, 210, 230, 255),
    )
    labels = (
        "[Performance]",
        "[3D Display]",
        "[Resolution]",
        "[Controller]",
        "[Environment]",
    )
    latency_text = f"{float(latency_ms):.0f}ms" if float(latency_ms or 0.0) > 0 else "N/A"
    values = (
        f"XR {float(actual_fps):.0f} FPS   SBS {float(sbs_fps):.0f} FPS   Latency {latency_text}",
        (
            f"{float(screen_width):.2f} x {float(screen_height):.2f} m"
            f"  @  {float(screen_distance):.2f} m"
            f"   Depth Strength {float(depth_strength):.2f}"
        ),
        f"XR {int(vr_res[0])}x{int(vr_res[1])}/eye   Screen {int(sbs_res[0])}x{int(sbs_res[1])}",
        f"Model: {controller_brand}" if controller_brand else "",
        "ON" if environment_visible else "OFF",
    )
    pad_x = 14.0
    pad_y = 14.0
    row_gap = 34.0
    label_width = max(
        atlas.text_advance(label, scale=scale) for label in labels
    )
    value_x = pad_x + label_width + 10.0
    value_width = max(
        atlas.text_advance(value, scale=scale) for value in values
    )
    canvas_width = int(math.ceil(value_x + value_width + pad_x))
    canvas_height = int(math.ceil(pad_y * 2.0 + row_gap * len(labels)))
    runs: list[dict[str, Any]] = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = pad_y + index * row_gap
        runs.append(
            {
                "text": label,
                "x": pad_x,
                "y": y,
                "scale": scale,
                "color": label_color,
            }
        )
        if value:
            runs.append(
                {
                    "text": value,
                    "x": value_x,
                    "y": y,
                    "scale": scale,
                    "color": value_colors[index],
                }
            )
    return _build_msdf_panel_request(
        atlas,
        tuple(runs),
        width=canvas_width,
        height=canvas_height,
        background=(32, 32, 36, 210),
    )


def _build_msdf_help_panel(
    atlas: MsdfFontAtlas,
    rows: list[tuple[str, str, str, bool]],
    *,
    two_columns: bool,
    size_scale: float = 1.0,
    canvas_scale: float | None = None,
) -> VulkanMsdfQuadRequest:
    """Build the controller guide panel from its shared row definition."""
    size_scale = max(0.1, min(4.0, float(size_scale)))
    normal_px = (16.0 if two_columns else 21.0) * size_scale
    title_px = (18.0 if two_columns else 21.0) * size_scale
    normal_scale = normal_px / max(float(atlas.line_height), 1.0)
    title_scale = title_px / max(float(atlas.line_height), 1.0)
    column_widths = [0.0, 0.0, 0.0]
    for row in rows:
        is_title = bool(row[3])
        scale = title_scale if is_title else normal_scale
        for column in range(3):
            column_widths[column] = max(
                column_widths[column],
                atlas.text_advance(row[column], scale=scale),
            )

    gap = 20.0 * size_scale
    middle_gap = 50.0 * size_scale
    padding_x = 30.0 * size_scale
    padding_y = 20.0 * size_scale
    line_height = (16.0 + 6.0 if two_columns else 21.0 + 6.0) * size_scale
    inner_width = sum(column_widths) + gap * 2.0
    if two_columns:
        title_indices = [index for index, row in enumerate(rows) if bool(row[3])]
        middle_index = title_indices[4] if len(title_indices) > 4 else len(rows)
        left_rows = rows[:middle_index]
        right_rows = rows[middle_index:]
    else:
        # The screen-side vertical guide is one complete column. Do not apply
        # the controller-attached two-column split to this layout.
        left_rows = rows
        right_rows = []
    content_width = (
        inner_width * 2.0 + middle_gap + padding_x * 2.0
        if two_columns
        else inner_width + padding_x * 2.0
    )
    content_height = max(len(left_rows), len(right_rows)) * line_height + padding_y * 2.0
    canvas_width = content_width
    canvas_height = content_height
    content_offset_x = 0.0
    content_offset_y = 0.0
    if canvas_scale is not None:
        canvas_scale = max(1.0, float(canvas_scale))
        base_normal_px = (16.0 if two_columns else 21.0) * canvas_scale
        base_title_px = (18.0 if two_columns else 21.0) * canvas_scale
        base_column_widths = [0.0, 0.0, 0.0]
        for row in rows:
            is_title = bool(row[3])
            base_row_scale = (
                base_title_px if is_title else base_normal_px
            ) / max(float(atlas.line_height), 1.0)
            for column in range(3):
                base_column_widths[column] = max(
                    base_column_widths[column],
                    atlas.text_advance(row[column], scale=base_row_scale),
                )
        base_gap = 20.0 * canvas_scale
        base_middle_gap = 50.0 * canvas_scale
        base_padding_x = 30.0 * canvas_scale
        base_padding_y = 20.0 * canvas_scale
        base_inner_width = sum(base_column_widths) + base_gap * 2.0
        base_line_height = (
            16.0 + 6.0 if two_columns else 21.0 + 6.0
        ) * canvas_scale
        canvas_width = (
            base_inner_width * 2.0 + base_middle_gap + base_padding_x * 2.0
            if two_columns
            else base_inner_width + base_padding_x * 2.0
        )
        canvas_height = (
            max(len(left_rows), len(right_rows)) * base_line_height
            + base_padding_y * 2.0
        )
        content_offset_x = (canvas_width - content_width) * 0.5
        content_offset_y = (canvas_height - content_height) * 0.5
    runs: list[dict[str, Any]] = []

    def add_rows(group_rows, origin_x: float) -> None:
        for row_index, row in enumerate(group_rows):
            is_title = bool(row[3])
            scale = title_scale if is_title else normal_scale
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            y = content_offset_y + padding_y + row_index * line_height
            x = origin_x
            for column in range(3):
                text = str(row[column])
                if text:
                    runs.append(
                        {
                            "text": text,
                            "x": content_offset_x + x,
                            "y": y,
                            "scale": scale,
                            "color": color,
                        }
                    )
                x += column_widths[column] + gap

    add_rows(left_rows, padding_x)
    if two_columns:
        add_rows(right_rows, padding_x + inner_width + middle_gap)
    return _build_msdf_panel_request(
        atlas,
        tuple(runs),
        width=int(math.ceil(canvas_width)),
        height=int(math.ceil(canvas_height)),
        background=(18, 18, 28, 210),
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
    headset_model: str = _DEFAULT_XR_HEADSET_PRESET.key
    controller_guide_max_distance: float = 0.4
    filament_bridge_path: str | None = None
    filament_glb_path: str | None = None
    filament_profile_path: str | None = None
    filament_scene_exposure_ev: float = 0.0
    filament_skybox_brightness: float = 1.0
    # Reuse the legacy headset preset for profiles without a screen. The
    # presenter only consumes these resolved values; it does not define a
    # second headset-geometry table.
    filament_screen_width: float = _DEFAULT_XR_HEADSET_PRESET.width_m
    filament_screen_distance: float = _DEFAULT_XR_HEADSET_PRESET.distance_m
    filament_ambient_light_color: tuple[float, float, float] = (0.14, 0.13, 0.15)
    filament_fill_light_color: tuple[float, float, float] = (0.55, 0.55, 0.58)
    filament_fill_light_intensity: float = 1.0
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
    array_size: int = 1


class OpenXrCompositionBuilder:
    """Builds projection layers without owning OpenXR frame lifecycle."""

    def __init__(self, xr: Any, reference_space: Any) -> None:
        self.xr = xr
        self.reference_space = reference_space

    def projection_layer(
        self, views: list[Any], swapchains: list[_EyeSwapchain]
    ) -> Any:
        layered = len(swapchains) == 1 and swapchains[0].array_size >= 2
        eye_swapchains = (
            [(swapchains[0], 0), (swapchains[0], 1)]
            if layered
            else [(eye, 0) for eye in swapchains]
        )
        if len(views) < len(eye_swapchains):
            raise ValueError("projection layer requires one view per eye swapchain")
        projection_views = []
        for eye_index, (eye, array_index) in enumerate(eye_swapchains):
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
                        image_array_index=array_index,
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
                image_array_index=eye_index if swapchain.array_size >= 2 else 0,
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
    CoreControllerGuideInputMixin,
    CoreControllerShortcutsMixin,
    CoreInputHelpersMixin,
):
    """OpenXR Vulkan projection-layer presenter with Filament controllers."""

    _VULKAN_EXTENSION = "XR_KHR_vulkan_enable2"

    def __init__(
        self,
        config: OpenXrVulkanConfig | None = None,
        *,
        on_headset_state: Callable[[str], None] | None = None,
        on_controller_shortcut: Callable[..., bool | None] | None = None,
        on_breakdown_inc: Callable[[str, int | float], None] | None = None,
        on_breakdown_add_time: Callable[[str, float], None] | None = None,
        on_breakdown_set_latest: Callable[[str, Any], None] | None = None,
        on_runtime_fps: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or OpenXrVulkanConfig()
        self._headset_preset = resolve_xr_headset_preset(self.config.headset_model)
        self._on_headset_state = on_headset_state
        self._on_controller_shortcut = on_controller_shortcut
        self._on_breakdown_inc = on_breakdown_inc
        self._on_breakdown_add_time = on_breakdown_add_time
        self._on_breakdown_set_latest = on_breakdown_set_latest
        self._on_runtime_fps = on_runtime_fps
        # The projection presenter does not submit an asynchronous effect
        # source. Mark that validation branch inactive instead of reporting a
        # missing effect path for every otherwise-valid projection frame.
        if self._on_breakdown_set_latest is not None:
            self._on_breakdown_set_latest("openxr_async_effects_enabled", False)
        if self.config.render_scale <= 0:
            raise ValueError("render_scale must be greater than zero")
        if len(self.config.clear_color) != 4:
            raise ValueError("clear_color must contain four components")
        if self.config.controller_guide_max_distance <= 0:
            raise ValueError("controller_guide_max_distance must be greater than zero")

        self.xr: Any = None
        self.instance: Any = None
        self.system_id: Any = None
        self.session: Any = None
        self.reference_space: Any = None
        self._reference_space_type: Any = None
        self.vulkan: VulkanContext | None = None
        self.swapchain_format: int | None = None
        self.swapchains: list[_EyeSwapchain] = []
        self._multiview_active = False
        self._vulkan_projection_composer_requested = _env_flag(
            "D2S_VULKAN_PROJECTION_COMPOSER", default=True
        )
        self._vulkan_projection_quality_chain_requested = _env_flag(
            "D2S_VULKAN_PROJECTION_QUALITY_CHAIN", default=True
        )
        self._vulkan_projection_composer_active = False
        self._vulkan_projection_composer_frame_id: int | None = None
        self._last_vulkan_projection_composer_status: tuple[Any, ...] | None = None
        self._last_vulkan_projection_composer_fallback: tuple[str, str] | None = None
        screen_quad_reprojection_requested = _env_flag(
            "D2S_OPENXR_SCREEN_QUAD_REPROJECTION"
        )
        self._screen_quad_reprojection_requested = bool(
            screen_quad_reprojection_requested
            and _SCREEN_QUAD_REPROJECTION_SUPPORTED
        )
        if (
            screen_quad_reprojection_requested
            and not _SCREEN_QUAD_REPROJECTION_SUPPORTED
        ):
            print(
                "[OpenXRViewer] Screen Quad Reprojection disabled: "
                "Virtual Desktop stereo Quad swapchain is unsupported; "
                "using Projection swapchain",
                flush=True,
            )
        self._screen_quad_reprojection_active = False
        self._screen_quad_reprojection_frame_id: int | None = None
        self._last_screen_quad_reprojection_status: tuple[Any, ...] | None = None
        if self._on_breakdown_set_latest is not None:
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_composer_requested",
                self._vulkan_projection_composer_requested,
            )
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_composer_active", False
            )
            self._on_breakdown_set_latest(
                "openxr_vulkan_projection_composer_frame_id", -1
            )
            self._on_breakdown_set_latest(
                "openxr_screen_quad_reprojection_requested",
                self._screen_quad_reprojection_requested,
            )
            self._on_breakdown_set_latest(
                "openxr_screen_quad_reprojection_active", False
            )
        self._quad_swapchains: list[_EyeSwapchain] = []
        self._quad_swapchain_format: int | None = None
        self._tool_quad_swapchain_format: int | None = None
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
        self._profile_alignment_logged = False
        self._head_position_w: np.ndarray | None = None
        self._head_forward_w: np.ndarray | None = None
        self._initial_head_y = 0.0
        self._profile_near_plane = 0.05
        self._profile_far_plane = 1000.0
        self._filament_scene_exposure = self.config.filament_scene_exposure_ev
        self._filament_skybox_brightness = self.config.filament_skybox_brightness
        self._filament_ambient_light_color = self.config.filament_ambient_light_color
        self._last_screen_resolution_status = None
        self._last_screen_sampling_status = None
        self._active_screen_sampling_plan: ScreenSamplingPlan | None = None
        self._controller_hdr_lighting = False
        self._filament_fill_light_color = self.config.filament_fill_light_color
        self._filament_fill_light_intensity = self.config.filament_fill_light_intensity
        self._filament_fill_light_direction = self.config.filament_fill_light_direction
        self._filament_lighting_presets: tuple[dict[str, Any], ...] = ()
        self._filament_lighting_preset_index = 0
        self._filament_glow_mode = "off"
        # Glow belongs to the blank Default environment only. GLB rooms have
        # authored lighting and geometry that must not receive this overlay.
        self._filament_glow_environment_enabled = not bool(
            self.config.filament_glb_path
        )
        # Keep the v2.5 effect constants intact. Glow samples its own small
        # Vulkan compute output, leaving the zero-copy screen image untouched.
        self._filament_glow_intensity = 0.175
        self._filament_glow_width = 0.75
        self._filament_glow_default_multiplier = 1.5
        self._filament_glow_intensity_multiplier = 0.0
        self._filament_glow_shell_default_multiplier = 1.85
        self._filament_glow_shell_intensity_multiplier = 0.0
        self._filament_glow_shell_radius = 20.0
        self._filament_glow_shell_height = 9.5
        self._frosted_glow_intensity = 1.0
        self._frosted_glow_alpha = 0.42
        self._frosted_glow_threshold = 0.46
        self._frosted_glow_lod = 5.4
        self._frosted_glow_blend = 1.35
        self._frosted_glow_thickness = 1.6
        self._frosted_glow_diffuse = 0.85
        self._frosted_glow_inset = 0.045
        self._frosted_veil_intensity = 1.5
        self._frosted_veil_alpha = 1.0
        self._last_filament_glow_source_serial = -1
        self._last_filament_glow_source_key: tuple[str, int] | None = None
        self._last_filament_glow_status: tuple[Any, ...] | None = None
        self._filament_screen: tuple[
            tuple[float, float, float], float, float, tuple[float, float, float]
        ] | None = None
        self._filament_screen_initial = None
        self._filament_screen_profile_authored = False
        self._filament_screen_head_initialized = False
        self._screen_curved = False
        self._passthrough_backdrop = False
        self._controllers_root = Path(__file__).resolve().parent / "controllers"
        self._controller_brands = discover_controller_brands(self._controllers_root)
        self._controller_brand = select_controller_brand(
            self._controller_brands,
            self.config.controller_model or os.environ.get("D2S_CONTROLLER_MODEL", "PICO"),
        )
        self._controller_calibration_mode = False
        self._controller_calibration_offset = np.asarray(
            self._controller_brand.offset if self._controller_brand else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        self._controller_calibration_rotation_deg = float(
            self._controller_brand.rotation_deg if self._controller_brand else 0.0
        )
        self._controller_b_button_local: np.ndarray | None = None
        self._controller_b_button_resolved = False
        self._controller_inputs = ({}, {})
        self._last_controller_input_error: str | None = None
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
        # Match the legacy laser edge-release cone: once the ray is within
        # six degrees of the nearest screen edge, keep the cursor attached.
        self._ray_edge_deadzone_rad = math.radians(6.0)
        self._ray_filter_l = OneEuroFilter3D(8.0, 8.0, 8.0)
        self._ray_filter_r = OneEuroFilter3D(8.0, 8.0, 8.0)
        self._last_frame_dt = 1.0 / 90.0
        self._initialized = False
        self._presenter_thread_id: int | None = None
        self._presenter_commands: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2)
        self._output_adapter: Any | None = None
        self._output_adapter_error: str | None = None
        self._next_output_frame_id = 0
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
        self._last_screen_quad_layers: list[Any] = []
        # One-shot first-frame visual diagnostics. The readback is deliberately
        # delayed until the normal render has completed, so it observes the
        # production Vulkan image and the final OpenXR projection target.
        self._visual_regression_capture_eyes: set[int] = set()
        self._visual_regression_capture_failed = False
        self._visual_regression_source_host_images: dict[int, VulkanHostImage] = {}
        self._visual_regression_projection_host_images: dict[int, VulkanHostImage] = {}
        self._overlay_quad_entries: dict[str, dict[str, Any]] = {}
        # Keep rasterized tool textures and their released swapchain image
        # alive. Static Quad layers must not perform a host upload every XR
        # frame; only the layer pose is rebuilt per frame.
        self._tool_quad_texture_cache: dict[str, np.ndarray] = {}
        self._tool_quad_texture_keys: dict[str, tuple[Any, ...]] = {}
        self._tool_overlay_xr_fps = 0.0
        self._tool_overlay_pending_xr_fps = 0.0
        self._tool_overlay_sbs_fps = 0.0
        self._tool_overlay_latency_ms = 0.0
        self._tool_overlay_depth_strength = 0.0
        self._tool_overlay_depth_strength_pending: float | None = None
        self._tool_overlay_vr_res = (0, 0)
        self._tool_overlay_sbs_res = (0, 0)
        self._tool_overlay_pending_latency_ms = 0.0
        self._tool_overlay_xr_window_started = 0.0
        self._tool_overlay_xr_window_frames = 0
        self._tool_overlay_xr_frame_ts = deque(maxlen=60)
        self._tool_overlay_sbs_window_started = 0.0
        self._tool_overlay_sbs_window_frames = 0
        self._tool_overlay_last_output_id: int | None = None
        self._right_grip_screen_pointer_applied = False
        self._controller_callout_rgba: np.ndarray | None = None
        self._msdf_font_atlas: MsdfFontAtlas | None = None
        self._vulkan_msdf_quad_renderer: VulkanMsdfQuadRenderer | None = None
        self._vulkan_projection_screen_pass: VulkanProjectionScreenPass | None = None
        # Legacy screen OSD state. These are rendered as Quad layers above
        # the virtual screen, never inside the projection scene.
        self._preset_name_overlay: str | None = None
        self._preset_osd_show_t = -999.0
        self._screen_osd_show_t = -999.0
        self._depth_osd_show_t = -999.0
        self._depth_osd_message: str | None = None
        # Legacy OpenXR shortcut state is kept in the presenter so both the
        # Vulkan projection path and future Quad Layer overlays read one state.
        self._keyboard_visible = False
        self._fps_overlay_visible = False
        self._operation_guide_visible = False
        self._screen_operation_guide_visible = False
        self._hand_fps_visible = False
        self._hand_operation_guide_visible = False
        self._aperture_visible = False
        self._init_controller_shortcuts()
        self._init_controller_guide_input()
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
        self._haptic_last_l = 0.0
        self._haptic_last_r = 0.0
        self._grip_l_now = False
        self._grip_r_now = False
        self._pointer_state = {"left": "idle", "right": "idle"}
        self._pointer_press_time = {"left": 0.0, "right": 0.0}
        self._left_grab_anchor = None
        self._right_grab_anchor = None
        self._screen_hit_grab_anchor_l = None
        self._screen_hit_grab_anchor_r = None
        self._keyboard_position_offset = np.zeros(3, dtype=np.float64)
        self._keyboard_rotation_offset = np.zeros(2, dtype=np.float64)
        self._keyboard_grab_anchor = None
        self._kb_grab_local_l = None
        self._kb_grab_local_r = None
        self._screen_resize_anchor = None
        self._grip_target_l = None
        self._grip_target_r = None
        self._grip_rotation_anchor_l = None
        self._grip_rotation_anchor_r = None
        self._screen_rotation_anchor_l = None
        self._screen_rotation_anchor_r = None
        self._grip_screen_rotation_snapped_l = False
        self._both_grip_anchor = None
        self._scroll_accum_x = 0.0
        self._scroll_accum_y = 0.0
        for direction in ("left", "right", "up", "down"):
            setattr(self, f"_arrow_{direction}_held", False)
        self._status_panel_cycle = 0
        self._hand_panel_cycle = 0
        self._unsupported_shortcut_actions: set[str] = set()
        default_screen_width = max(0.25, float(self.config.filament_screen_width))
        default_screen_distance = max(0.25, float(self.config.filament_screen_distance))
        self._shortcut_screen_presets = (
            ('10" Tablet', 0.30, 0.4),
            ('27" Monitor', 0.60, 0.6),
            ('65" TV', 1.44, 2.0),
            ('100" Projector 1', 2.40, 2.0),
            ('100" Projector 2', 2.21, 2.5),
            ('Headset Recommended', default_screen_width, default_screen_distance),
            ('1000" IMAX', 22.0, 20.0),
        )
        self._shortcut_screen_preset_index = 5
        self._shortcut_saved_skybox_brightness = self._filament_skybox_brightness
        self._shortcut_light_levels = (0.0, 0.5, 1.0)
        # Right-grip screen controls accelerate while the stick is held. The
        # first frame remains precise, then reaches 10 m/s after five seconds.
        self._screen_control_min_speed = 0.10
        self._screen_control_max_speed = 10.0
        self._screen_control_acceleration = (
            self._screen_control_max_speed - self._screen_control_min_speed
        ) / 5.0
        self._screen_control_max_hold_seconds = 5.0
        self._screen_distance_hold_seconds = 0.0
        self._screen_distance_hold_direction = 0
        self._screen_size_hold_seconds = 0.0
        self._screen_size_hold_direction = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def output_ready(self) -> bool:
        """Report readiness after the presenter-owned Filament Engine is ready."""
        return self._initialized

    @property
    def source_ready_semaphore_available(self) -> bool:
        """Whether Projection Composer can consume exported source semaphores."""
        return bool(self._vulkan_projection_composer_requested and self.vulkan is not None)

    def _controller_ambient_light_color(self) -> tuple[float, float, float]:
        """Return the room ambient color without controller compensation."""
        return tuple(float(component) for component in self._filament_ambient_light_color)

    def _controller_hdr_ambient_light_color(self) -> tuple[float, float, float]:
        """Return controller ambient light isolated from the room Scene."""
        multiplier = max(
            0.0,
            float(getattr(self._controller_brand, "ambient_light_multiplier", 1.0)),
        )
        return tuple(
            float(component) * multiplier
            for component in self._filament_ambient_light_color
        )

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
            self._report_projection_composer_boundary()
            self._xr_instance = self.instance
            self._xr_session = self.session
            self._xr_space = self.reference_space
            self._init_controller_actions()
            self._load_filament_profile()
            # Filament Engine and all resources remain owned by the Presenter
            # thread. Filament rejects rendering from a thread that was not
            # adopted by its JobSystem, so native GLB loading cannot migrate
            # the Engine to a background thread.
            self._initialize_filament_bridges()
            self._initialize_msdf_text_atlas()
            self._initialize_msdf_quad_renderer()
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
            elif event.type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                # The legacy viewer rebuilt its base reference space and then
                # reapplied the authored seat/profile transform. Keep the
                # same two-step contract so controller poses, screen poses,
                # and projection views share one calibrated space.
                self._recreate_reference_space_after_runtime_change()
            elif event.type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self.exit_requested = True

    def _recreate_reference_space_after_runtime_change(self) -> None:
        """Recreate the base XR space after a runtime relocation event."""
        if self.xr is None or self.session is None or self._reference_space_type is None:
            self._profile_space_applied = False
            return
        try:
            new_space = self.xr.create_reference_space(
                self.session,
                self.xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._reference_space_type
                ),
            )
        except Exception as exc:
            # Keep the current space alive if the runtime cannot create a
            # replacement; the next valid view will retry profile application.
            print(
                f"[OpenXRViewer] Reference space change pending; "
                f"recreate failed: {exc}",
                flush=True,
            )
            self._profile_space_applied = False
            return
        old_space = self.reference_space
        self.reference_space = new_space
        self._xr_space = new_space
        self._profile_space_applied = False
        self._profile_initial_head = None
        self._profile_alignment_logged = False
        self._head_position_w = None
        self._head_forward_w = None
        if old_space is not None:
            try:
                self.xr.destroy_space(old_space)
            except Exception:
                pass

    def run_frame(self) -> bool:
        frame_started = time.perf_counter()
        self._ensure_initialized()
        events_started = time.perf_counter()
        self.poll_events()
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_events", time.perf_counter() - events_started
            )
        if self.exit_requested:
            return False
        if not self.session_running:
            self._notify_headset_waiting()
            time.sleep(0.01)
            return True

        xr = self.xr
        wait_started = time.perf_counter()
        frame_state = xr.wait_frame(self.session)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_wait_frame", time.perf_counter() - wait_started
            )
        # Keep xrWaitFrame at the frame boundary. Runtime-output conversion can
        # enqueue Vulkan work and must not delay the runtime's pacing decision.
        commands_started = time.perf_counter()
        self._drain_presenter_commands()
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_presenter_commands", time.perf_counter() - commands_started
            )
        if self._on_breakdown_inc is not None:
            self._on_breakdown_inc("openxr_loop", 1)
            self._on_breakdown_inc(
                "openxr_should_render" if frame_state.should_render else "openxr_no_render",
                1,
            )
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
        controls_started = time.perf_counter()
        try:
            if self._on_breakdown_inc is not None:
                self._on_breakdown_inc("openxr_input_sample", 1)
            self._sync_controller_inputs(1.0 / 90.0)
            self._update_aim_poses(frame_state.predicted_display_time)
            self._update_grip_poses(frame_state.predicted_display_time)
            self._smooth_controller_poses()
            self._grip_l_now = bool(self._controller_input(0).get("grip", 0.0) > 0.5)
            self._grip_r_now = bool(self._controller_input(1).get("grip", 0.0) > 0.5)
            self._handle_keyboard_input()
            self._handle_vulkan_pointer_input()
            self._handle_controller_shortcuts()
            self._handle_controller_guide_input(self._last_frame_dt)
            self._last_controller_input_error = None
        except Exception as exc:
            # Keep one bad optional input path from terminating XR, but make
            # the failure observable instead of silently disabling all
            # keyboard, drag, and shortcut handling for the session.
            error = f"{type(exc).__name__}: {exc}"
            if error != self._last_controller_input_error:
                self._last_controller_input_error = error
                if type(exc).__name__ == "SessionNotFocused":
                    message = (
                        "[OpenXRViewer] Controller input deferred: "
                        "OpenXR session is not focused"
                    )
                else:
                    message = f"[OpenXRViewer] Controller input update failed: {error}"
                print(
                    message,
                    flush=True,
                )
        finally:
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_controls", time.perf_counter() - controls_started
                )
        begin_started = time.perf_counter()
        xr.begin_frame(self.session)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_begin_frame", time.perf_counter() - begin_started
            )
        layer_structures: list[Any] = []
        layer_pointers: list[Any] = []
        try:
            if frame_state.should_render:
                locate_started = time.perf_counter()
                locate_call_started = time.perf_counter()
                view_state, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(
                        view_configuration_type=self._view_configuration_type,
                        display_time=frame_state.predicted_display_time,
                        space=self.reference_space,
                    ),
                )
                if self._on_breakdown_add_time is not None:
                    self._on_breakdown_add_time(
                        "openxr_locate_call", time.perf_counter() - locate_call_started
                    )
                valid_flags = (
                    xr.ViewStateFlags.POSITION_VALID_BIT
                    | xr.ViewStateFlags.ORIENTATION_VALID_BIT
                )
                if view_state.view_state_flags & valid_flags == valid_flags:
                    if self._apply_profile_reference_space(views):
                        locate_call_started = time.perf_counter()
                        view_state, views = xr.locate_views(
                            self.session,
                            xr.ViewLocateInfo(
                                view_configuration_type=self._view_configuration_type,
                                display_time=frame_state.predicted_display_time,
                                space=self.reference_space,
                            ),
                        )
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_locate_call",
                                time.perf_counter() - locate_call_started,
                            )
                    view_prepare_started = time.perf_counter()
                    self._cache_head_position(views)
                    self._report_profile_alignment()
                    self._initialize_filament_screen_from_head()
                    if self._on_breakdown_add_time is not None:
                        self._on_breakdown_add_time(
                            "openxr_view_prepare",
                            time.perf_counter() - view_prepare_started,
                        )
                    output_lock_started = time.perf_counter()
                    with self._output_lock:
                        output_frame = self._pending_output
                    if self._on_breakdown_add_time is not None:
                        self._on_breakdown_add_time(
                            "openxr_output_lock",
                            time.perf_counter() - output_lock_started,
                        )
                    metrics_started = time.perf_counter()
                    self._update_tool_overlay_metrics(output_frame)
                    if self._on_breakdown_add_time is not None:
                        self._on_breakdown_add_time(
                            "openxr_overlay_metrics",
                            time.perf_counter() - metrics_started,
                        )
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
                        projection_started = time.perf_counter()
                        layer = self._render_projection_layer(views, output_frame)
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_projection_layer",
                                time.perf_counter() - projection_started,
                            )
                    if layer is not None:
                        layer_assembly_started = time.perf_counter()
                        layer_structures.append(layer)
                        layer_pointers.append(ctypes.pointer(layer))
                        try:
                            quad_started = time.perf_counter()
                            self._last_quad_layers = self._render_quad_layers(output_frame)
                            if self._on_breakdown_add_time is not None:
                                self._on_breakdown_add_time(
                                    "openxr_quad_total",
                                    time.perf_counter() - quad_started,
                                )
                            if (
                                output_frame is not None
                                and not self._screen_quad_reprojection_active
                            ):
                                commit_started = time.perf_counter()
                                self._commit_output_frame(output_frame)
                                if self._on_breakdown_add_time is not None:
                                    self._on_breakdown_add_time(
                                        "openxr_output_commit",
                                        time.perf_counter() - commit_started,
                                    )
                        except Exception:
                            if output_frame is not None:
                                self._abort_output_frame(output_frame)
                            raise
                        self._has_presented_frame = True
                        layer_pointers_started = time.perf_counter()
                        layer_structures.extend(self._last_quad_layers)
                        layer_pointers.extend(
                            ctypes.pointer(item) for item in self._last_quad_layers
                        )
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_layer_pointers",
                                time.perf_counter() - layer_pointers_started,
                            )
                        if self._on_breakdown_add_time is not None:
                            self._on_breakdown_add_time(
                                "openxr_layer_assembly",
                                time.perf_counter() - layer_assembly_started,
                            )
                if self._on_breakdown_add_time is not None:
                    self._on_breakdown_add_time(
                        "openxr_locate_total", time.perf_counter() - locate_started
                    )
        finally:
            if not bool(getattr(self.vulkan, "device_lost", False)):
                end_info = xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=self._environment_blend_mode,
                    layer_count=len(layer_pointers),
                    layers=layer_pointers or None,
                )
                end_started = time.perf_counter()
                xr.end_frame(self.session, end_info)
                self._record_xr_presented_frame()
                if self._on_breakdown_add_time is not None:
                    self._on_breakdown_add_time(
                        "openxr_end_frame", time.perf_counter() - end_started
                    )
        self.frame_count += 1
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_frame_total", time.perf_counter() - frame_started
            )
        return not self.exit_requested

    def _set_shortcut_panel(self, name: str | None) -> None:
        # Legacy Menu/A cycle: hidden -> FPS -> FPS + vertical screen guide
        # -> hidden. The guide never replaces the FPS panel at state 2.
        # Menu and B panels are mutually exclusive so a stale guide cannot
        # remain visible when the user switches to the other control path.
        self._hand_panel_cycle = 0
        self._hand_fps_visible = False
        self._hand_operation_guide_visible = False
        self._fps_overlay_visible = name in {"fps", "guide"}
        self._screen_operation_guide_visible = name == "guide"
        self._aperture_visible = name == "aperture"
        self._operation_guide_visible = self._screen_operation_guide_visible

    def _set_hand_shortcut_panel(self, name: str | None) -> None:
        # Legacy B cycle: hidden -> hand FPS -> hand FPS + hand guide -> hidden.
        # Selecting the B panel clears the Menu-owned screen panel first.
        self._status_panel_cycle = 0
        self._fps_overlay_visible = False
        self._screen_operation_guide_visible = False
        self._aperture_visible = False
        self._hand_fps_visible = name in {"fps", "guide"}
        self._hand_operation_guide_visible = name == "guide"
        # Keep the legacy compatibility flag true for the controller-attached
        # B-panel; Menu uses _screen_operation_guide_visible above.
        self._operation_guide_visible = (
            self._screen_operation_guide_visible or self._hand_operation_guide_visible
        )

    def _set_shortcut_skybox_brightness(self, brightness: float) -> None:
        self._filament_skybox_brightness = max(0.0, float(brightness))
        if self.filament_bridge is not None:
            self.filament_bridge.set_skybox_brightness(
                self._filament_skybox_brightness
            )

    @staticmethod
    def _normalize_filament_glow_mode(value: Any) -> str:
        mode = str(value or "off").strip().lower()
        return {
            "none": "off",
            "false": "off",
            "0": "off",
            "screen": "glow",
            "frost": "frosted",
            "frost_glow": "frosted",
            "frosted_glow": "frosted",
        }.get(mode, mode) if mode in {
            "off", "none", "false", "0", "screen", "surround",
            "glow", "glow2", "veil", "frost", "frost_glow",
            "frosted", "frosted_glow",
        } else "off"

    def _apply_filament_glow_profile_fields(self, values: dict[str, Any]) -> None:
        if "glow_mode" in values:
            self._filament_glow_mode = self._normalize_filament_glow_mode(
                values.get("glow_mode")
            )
        for key, attribute, minimum, maximum in (
            ("glow_intensity", "_filament_glow_intensity", 0.0, None),
            ("glow_width", "_filament_glow_width", 0.0, None),
            ("glow_intensity_multiplier", "_filament_glow_intensity_multiplier", 0.0, None),
            ("glow_shell_intensity_multiplier", "_filament_glow_shell_intensity_multiplier", 0.0, None),
            ("glow_shell_radius", "_filament_glow_shell_radius", 0.0, None),
            ("glow_shell_height", "_filament_glow_shell_height", 0.0, None),
            ("frosted_glow_intensity", "_frosted_glow_intensity", 0.0, None),
            ("frosted_glow_alpha", "_frosted_glow_alpha", 0.0, 1.0),
            ("frosted_glow_threshold", "_frosted_glow_threshold", 0.0, 1.0),
            ("frosted_glow_lod", "_frosted_glow_lod", 0.0, None),
            ("frosted_glow_blend", "_frosted_glow_blend", 0.0, None),
            ("frosted_glow_thickness", "_frosted_glow_thickness", 0.1, None),
            ("frosted_glow_diffuse", "_frosted_glow_diffuse", 0.0, None),
            ("frosted_glow_inset", "_frosted_glow_inset", 0.0, None),
            ("frosted_veil_intensity", "_frosted_veil_intensity", 0.0, None),
            ("frosted_veil_alpha", "_frosted_veil_alpha", 0.0, 1.0),
        ):
            if key not in values:
                continue
            try:
                number = max(float(minimum), float(values[key]))
                if maximum is not None:
                    number = min(float(maximum), number)
                setattr(self, attribute, number)
            except (TypeError, ValueError):
                continue

    def _cycle_filament_glow_mode(self) -> None:
        modes = ("surround", "glow", "glow2", "veil", "frosted", "off")
        current = self._normalize_filament_glow_mode(self._filament_glow_mode)
        if current not in modes:
            current = (
                "glow"
                if float(self._filament_glow_intensity_multiplier) > 0.0
                else "off"
            )
        next_mode = modes[(modes.index(current) + 1) % len(modes)]
        self._filament_glow_mode = next_mode
        if next_mode == "off":
            self._filament_glow_intensity_multiplier = 0.0
            self._filament_glow_shell_intensity_multiplier = 0.0
        elif next_mode == "surround":
            self._filament_glow_intensity_multiplier = 0.0
            if self._filament_glow_shell_intensity_multiplier <= 0.0:
                self._filament_glow_shell_intensity_multiplier = (
                    self._filament_glow_shell_default_multiplier
                )
        else:
            self._filament_glow_shell_intensity_multiplier = 0.0
        if (
            next_mode not in {"off", "surround"}
            and self._filament_glow_intensity_multiplier <= 0.0
        ):
            self._filament_glow_intensity_multiplier = (
                self._filament_glow_default_multiplier
            )
        label = {
            "surround": "Surround Glow",
            "glow": "Glow",
            "glow2": "Glow2",
            "veil": "Veil",
            "frosted": "Frosted",
            "off": "Off",
        }[next_mode]
        self._preset_name_overlay = label
        self._preset_osd_show_t = time.perf_counter()
        self._last_filament_glow_status = None
        print(f"[OpenXRViewer] Glow mode: {next_mode}", flush=True)

    def _apply_filament_lighting_preset(
        self, preset: dict[str, Any], *, apply_bridge: bool = True
    ) -> None:
        """Apply the legacy environment lighting-preset fields to Filament."""
        if not isinstance(preset, dict):
            return
        for key, attribute in (
            ("preview_exposure", "_filament_scene_exposure"),
            ("env_exposure", "_filament_scene_exposure"),
            ("preview_skybox_brightness", "_filament_skybox_brightness"),
            ("controller_head_light_intensity", "_filament_fill_light_intensity"),
        ):
            if key in preset:
                try:
                    setattr(self, attribute, float(preset[key]))
                except (TypeError, ValueError):
                    pass
        for keys, attribute in (
            (("env_ambient_color", "ambient_color"), "_filament_ambient_light_color"),
            (("env_head_light_color", "head_light_color"), "_filament_fill_light_color"),
        ):
            for key in keys:
                value = preset.get(key)
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    try:
                        setattr(self, attribute, tuple(float(item) for item in value[:3]))
                    except (TypeError, ValueError):
                        pass
                    break
        direction = preset.get("env_fill_light_direction", preset.get("fill_light_direction"))
        if isinstance(direction, (list, tuple)) and len(direction) >= 3:
            try:
                self._filament_fill_light_direction = tuple(
                    float(item) for item in direction[:3]
                )
            except (TypeError, ValueError):
                pass
        self._apply_filament_glow_profile_fields(preset)
        if not apply_bridge or self.filament_bridge is None:
            return
        bridge = self.filament_bridge
        bridge.set_scene_exposure(self._filament_scene_exposure)
        bridge.set_skybox_brightness(self._filament_skybox_brightness)
        if hasattr(bridge, "set_ambient_light"):
            bridge.set_ambient_light(self._controller_ambient_light_color())
        if hasattr(bridge, "set_controller_ambient_light"):
            bridge.set_controller_ambient_light(
                self._controller_hdr_ambient_light_color(),
                True,
            )
        bridge.set_fill_light(
            self._filament_fill_light_color,
            self._filament_fill_light_intensity,
            self._filament_fill_light_direction,
        )

    def _cycle_shortcut_screen_preset(self) -> None:
        if self._filament_screen is None:
            return
        self._shortcut_screen_preset_index = (
            self._shortcut_screen_preset_index + 1
        ) % len(self._shortcut_screen_presets)
        self._apply_shortcut_screen_preset(self._shortcut_screen_preset_index)

    def _apply_shortcut_screen_preset(self, index: int) -> None:
        """Apply the legacy screen preset size, distance, and head-facing pose."""
        if self._filament_screen is None or not self._shortcut_screen_presets:
            return
        index = int(index) % len(self._shortcut_screen_presets)
        _name, width, distance = self._shortcut_screen_presets[index]
        old_position, old_width, old_height, rotation = self._filament_screen
        if self._head_position_w is not None and self._head_forward_w is not None:
            hx, _hy, hz = self._head_position_w
            fx, _fy, fz = self._head_forward_w
            horizontal = math.sqrt(float(fx) * float(fx) + float(fz) * float(fz))
            if horizontal > 1e-4:
                fx /= horizontal
                fz /= horizontal
            else:
                fx, fz = 0.0, -1.0
            position = (
                float(hx) + float(fx) * float(distance),
                float(self._initial_head_y),
                float(hz) + float(fz) * float(distance),
            )
            rotation = (
                math.degrees(math.atan2(-float(fx), -float(fz))),
                0.0,
                0.0,
            )
        else:
            position = (0.0, 0.0, -float(distance))
            rotation = (0.0, 0.0, 0.0)
        height = float(width) * float(old_height) / max(float(old_width), 1e-6)
        self._filament_screen = (
            tuple(float(value) for value in position),
            float(width),
            height,
            rotation,
        )
        self._preset_name_overlay = (
            f"{_name}  {float(width):.2f} x {float(height):.2f} m"
            f"  @ {float(distance):.2f} m"
        )
        self._preset_osd_show_t = time.perf_counter()

    def _controller_callback_depth_strength(self) -> float | None:
        """Read the synchronously updated runtime value when available."""
        callback = self._on_controller_shortcut
        owner = getattr(callback, "__self__", None)
        context = getattr(owner, "context", None)
        state = getattr(context, "openxr_state", None)
        snapshot = getattr(state, "runtime_settings_snapshot", None)
        value = getattr(snapshot, "depth_strength", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return max(0.0, value)

    def _dispatch_controller_shortcut(self, action: str, **values) -> None:
        """Apply shared shortcut actions to Vulkan-owned presentation state."""
        if action == "cycle_status_panel":
            self._status_panel_cycle = (self._status_panel_cycle + 1) % 3
            self._set_shortcut_panel(
                (None, "fps", "guide")[self._status_panel_cycle]
            )
        elif action == "cycle_hand_panel":
            # Match the legacy B long-press state machine exactly.
            self._hand_panel_cycle = (self._hand_panel_cycle + 1) % 3
            self._set_hand_shortcut_panel(
                (None, "fps", "guide")[self._hand_panel_cycle]
            )
        elif action == "toggle_keyboard":
            self._keyboard_visible = not self._keyboard_visible
            self._keyboard_position_offset[:] = 0.0
            self._keyboard_grab_anchor = None
            self._kb_grab_local_l = None
            self._kb_grab_local_r = None
            if self._keyboard_visible:
                screen_width = float(self._filament_screen[1]) if self._filament_screen else 2.4
                self._keyboard_width = max(0.3, screen_width * 0.8)
                self._keyboard_height = self._keyboard_width * _KB_TEX_H / float(_KB_TEX_W)
                self._keyboard_keys = []
                self._keyboard_texture_key = None
        elif action == "reset_screen":
            if self._filament_screen_profile_authored:
                if self._filament_screen_initial is not None:
                    self._filament_screen = self._filament_screen_initial
                    self._preset_name_overlay = "Screen Reset"
                    self._preset_osd_show_t = time.perf_counter()
            else:
                self._shortcut_screen_preset_index = 5
                self._apply_shortcut_screen_preset(5)
        elif action == "cycle_screen_preset":
            self._cycle_shortcut_screen_preset()
        elif action == "toggle_screen_shape":
            self._screen_curved = not self._screen_curved
            self._preset_name_overlay = (
                "Curved Screen" if self._screen_curved else "Flat Screen"
            )
            self._preset_osd_show_t = time.perf_counter()
        elif action == "toggle_background":
            if self._filament_skybox_brightness > 0.0:
                self._shortcut_saved_skybox_brightness = (
                    self._filament_skybox_brightness
                )
                self._set_shortcut_skybox_brightness(0.0)
            else:
                self._set_shortcut_skybox_brightness(
                    self._shortcut_saved_skybox_brightness or 1.0
                )
        elif action == "cycle_environment_light":
            # The shared shortcut name predates the renderer split. In v2.5,
            # releasing X after 1-4 seconds cycles the screen-edge effects;
            # it does not cycle room-light presets.
            self._cycle_filament_glow_mode()
        elif action == "toggle_passthrough":
            bridge = self.filament_bridge
            if bridge is None or not getattr(
                bridge, "passthrough_backdrop_abi_available", False
            ):
                self._unsupported_shortcut_actions.add(action)
                return
            self._passthrough_backdrop = not self._passthrough_backdrop
            bridge.set_passthrough_backdrop(self._passthrough_backdrop)
        elif action == "switch_controller_brand":
            self._switch_shortcut_controller_brand()
        elif action == "toggle_controller_calibration":
            self._controller_calibration_mode = not self._controller_calibration_mode
            print(
                "[OpenXRViewer] Controller calibration: "
                f"{'on' if self._controller_calibration_mode else 'off'}",
                flush=True,
            )
        elif action == "adjust_controller_calibration":
            self._controller_calibration_offset[1] += float(values.get("offset_y", 0.0))
            self._controller_calibration_offset[2] += float(values.get("offset_z", 0.0))
            self._controller_calibration_rotation_deg += float(
                values.get("rotation_deg", 0.0)
            )
        elif action == "save_controller_calibration":
            self._save_shortcut_controller_calibration()
        elif action == "rotate_screen":
            if self._screen_ray_hit_for_hand(0) is not None:
                self._adjust_shortcut_screen_rotation(
                    float(values.get("yaw_delta", 0.0)),
                    float(values.get("pitch_delta", 0.0)),
                )
        elif action == "resize_screen":
            # The pointer path below already applies the legacy exponential
            # right-grip/right-stick curve. Do not add the guide mixin's old
            # fixed-speed delta a second time in the same XR frame.
            if (
                not self._right_grip_screen_pointer_applied
                and self._screen_ray_hit_for_hand(1) is not None
            ):
                self._adjust_shortcut_screen_size(
                    float(values.get("width_delta", 0.0)),
                    float(values.get("distance_delta", 0.0)),
                )
        elif action == "rotate_keyboard":
            self._keyboard_rotation_offset += np.asarray(
                (values.get("yaw_delta", 0.0), values.get("pitch_delta", 0.0)),
                dtype=np.float64,
            )
        elif action == "orbit_keyboard":
            self._keyboard_position_offset[0] += float(values.get("horizontal", 0.0)) * 0.4
            self._keyboard_position_offset[1] += float(values.get("vertical", 0.0)) * 0.4
        elif action == "resize_keyboard":
            self._adjust_shortcut_keyboard(
                float(values.get("width_delta", 0.0)),
                float(values.get("distance_delta", 0.0)),
            )
        elif action == "arrow_axes":
            self._send_arrow_impl(float(values.get("horizontal", 0.0)), "left", "right")
            self._send_arrow_impl(float(values.get("vertical", 0.0)), "up", "down")
        elif action == "scroll_axes":
            self._accum_scroll(
                float(values.get("horizontal", 0.0)),
                float(values.get("vertical", 0.0)),
                float(values.get("dt", self._last_frame_dt)),
            )
        elif action == "copy":
            _send_key(0x43, ctrl=True)
        elif action == "cut":
            _send_key(0x58, ctrl=True)
        elif action == "paste":
            _send_key(0x56, ctrl=True)
        elif action == "enter":
            _send_key(0x0D)
        else:
            handled = bool(
                self._on_controller_shortcut
                and self._on_controller_shortcut(action, **values)
            )
            if handled and action in {
                "adjust_depth_strength",
                "reset_depth",
                "toggle_stereo",
            }:
                # RuntimeCallbacks owns the depth value. The presenter owns
                # the legacy Quad prompt. Read the callback's synchronous
                # runtime snapshot so continuous stick input updates the
                # displayed value before the next rendered output arrives.
                callback_depth = self._controller_callback_depth_strength()
                previous_depth = self._tool_overlay_depth_strength
                if callback_depth is not None:
                    self._tool_overlay_depth_strength = callback_depth
                    self._tool_overlay_depth_strength_pending = callback_depth
                elif action == "adjust_depth_strength":
                    target = max(
                        0.0,
                        min(
                            10.0,
                            previous_depth + float(values.get("delta", 0.0)),
                        ),
                    )
                    self._tool_overlay_depth_strength = target
                    self._tool_overlay_depth_strength_pending = target
                if action == "toggle_stereo":
                    was_enabled = previous_depth > 0.0
                    self._depth_osd_message = (
                        "3D mode off"
                        if (callback_depth == 0.0 or (callback_depth is None and was_enabled))
                        else "3D mode on"
                    )
                else:
                    self._depth_osd_message = None
                self._depth_osd_show_t = time.perf_counter()
            if not handled:
                self._unsupported_shortcut_actions.add(action)

    def _input_deadzone(self) -> float:
        return 0.15

    def _adjust_shortcut_screen_rotation(
        self, yaw_delta: float, pitch_delta: float
    ) -> None:
        if self._filament_screen is None:
            return
        position, width, height, rotation = self._filament_screen
        next_rotation = (
            float(rotation[0]) + yaw_delta,
            max(-89.0, min(89.0, float(rotation[1]) + pitch_delta)),
            float(rotation[2]),
        )
        self._filament_screen = (position, width, height, next_rotation)
        self._screen_osd_show_t = time.perf_counter()

    def _adjust_shortcut_screen_size(
        self, width_delta: float, distance_delta: float
    ) -> None:
        if self._filament_screen is None:
            return
        position, width, height, rotation = self._filament_screen
        next_width = max(0.3, float(width) + width_delta)
        next_height = next_width * float(height) / max(float(width), 1e-6)
        head = np.asarray(
            self._head_position_w if self._head_position_w is not None else (0, 0, 0),
            dtype=np.float64,
        )
        radial = np.asarray(position, dtype=np.float64) - head
        distance = max(float(np.linalg.norm(radial)), 1e-6)
        next_distance = max(0.3, distance + distance_delta)
        next_position = head + radial / distance * next_distance
        self._filament_screen = (
            tuple(float(value) for value in next_position),
            next_width,
            next_height,
            rotation,
        )
        self._screen_osd_show_t = time.perf_counter()

    def _adjust_shortcut_keyboard(
        self, width_delta: float, distance_delta: float
    ) -> None:
        self._keyboard_width = max(0.3, min(4.0, self._keyboard_width + width_delta))
        self._keyboard_height = self._keyboard_width * _KB_TEX_H / float(_KB_TEX_W)
        self._keyboard_texture_key = None
        pose = self._keyboard_pose_mat4()
        head = np.asarray(
            self._head_position_w if self._head_position_w is not None else (0, 0, 0),
            dtype=np.float64,
        )
        radial = pose[:3, 3].astype(np.float64) - head
        distance = max(float(np.linalg.norm(radial)), 1e-6)
        self._keyboard_position_offset += radial / distance * distance_delta

    def _switch_shortcut_controller_brand(self) -> None:
        if not self._controller_brands:
            return
        names = sorted(self._controller_brands)
        current_name = getattr(self._controller_brand, "name", None)
        index = names.index(current_name) if current_name in names else -1
        next_brand = self._controller_brands[names[(index + 1) % len(names)]]
        previous = self._controller_brand
        bridge = self.filament_bridge
        try:
            if bridge is not None and hasattr(bridge, "load_controller"):
                bridge.load_controller(0, next_brand.left_glb.read_bytes())
                bridge.load_controller(1, next_brand.right_glb.read_bytes())
        except Exception:
            if bridge is not None and previous is not None:
                bridge.load_controller(0, previous.left_glb.read_bytes())
                bridge.load_controller(1, previous.right_glb.read_bytes())
            raise
        self._controller_brand = next_brand
        self._controller_calibration_offset = np.asarray(
            next_brand.offset, dtype=np.float64
        )
        self._controller_calibration_rotation_deg = float(next_brand.rotation_deg)
        ambient_multiplier = float(
            getattr(next_brand, "ambient_light_multiplier", 1.0)
        )
        if bridge is not None and hasattr(bridge, "set_ambient_light"):
            bridge.set_ambient_light(self._controller_ambient_light_color())
        if bridge is not None and hasattr(bridge, "set_controller_ambient_light"):
            bridge.set_controller_ambient_light(
                self._controller_hdr_ambient_light_color(),
                True,
            )
        anchor = self._resolve_controller_b_button_local(force=True)
        anchor_text = (
            "unresolved"
            if anchor is None
            else ", ".join(f"{value:.6f}" for value in anchor)
        )
        print(
            f"[OpenXRViewer] Switched controller: {next_brand.name}; "
            f"ambient_multiplier={ambient_multiplier:.2f}; "
            f"B-button anchor=({anchor_text})",
            flush=True,
        )

    def _save_shortcut_controller_calibration(self) -> None:
        brand = self._controller_brand
        if brand is None:
            return
        profile_path = brand.root / "profile.json"
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            profile = {}
        overrides = profile.setdefault("overrides", {})
        overrides["model_offset"] = [
            round(float(value), 6) for value in self._controller_calibration_offset
        ]
        overrides["model_rotation_deg"] = round(
            float(self._controller_calibration_rotation_deg), 4
        )
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._controller_calibration_mode = False
        print(f"[OpenXRViewer] Controller calibration saved: {profile_path}", flush=True)

    def _pulse_haptic(
        self,
        hand_path_str="/user/hand/right",
        *,
        amplitude=0.18,
        duration_s=0.018,
        min_interval_s=0.045,
    ) -> bool:
        """Send a short controller haptic pulse; failures are non-fatal."""
        action = getattr(self, "_act_haptic", None)
        xr = getattr(self, "xr", None)
        session = getattr(self, "session", None)
        if session is None:
            session = getattr(self, "_xr_session", None)
        if action is None or xr is None or session is None:
            return False

        now = time.perf_counter()
        last_attr = (
            "_haptic_last_l"
            if hand_path_str == "/user/hand/left"
            else "_haptic_last_r"
        )
        if now - float(getattr(self, last_attr, 0.0) or 0.0) < float(min_interval_s):
            return False

        try:
            path = getattr(
                self,
                "_path_left" if hand_path_str == "/user/hand/left" else "_path_right",
                None,
            )
            if path is None:
                instance = getattr(self, "instance", None)
                if instance is None:
                    instance = getattr(self, "_xr_instance", None)
                if instance is None:
                    return False
                path = xr.string_to_path(instance, hand_path_str)
            duration_ns = max(1, int(float(duration_s) * 1_000_000_000))
            vibration = xr.HapticVibration(
                duration=duration_ns,
                frequency=xr.FREQUENCY_UNSPECIFIED,
                amplitude=max(0.0, min(1.0, float(amplitude))),
            )
            xr.apply_haptic_feedback(
                session,
                xr.HapticActionInfo(action=action, subaction_path=path),
                vibration,
            )
            setattr(self, last_attr, now)
            return True
        except Exception:
            return False

    def _press_key(self, key, key_idx, held_key_attr, held_mods_attr):
        return self._press_key_impl(key, key_idx, held_key_attr, held_mods_attr)

    def _refresh_or_upload_keyboard_content(self) -> None:
        # Tool quads rebuild their RGBA payload from the current state each XR tick.
        return None

    def _adjust_frosted_glow_vk(self, _vk_code: int) -> bool:
        return False

    def _keyboard_pose_mat4(self) -> np.ndarray:
        _position, _screen_width, screen_height, _rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        screen_pose = self._filament_screen_pose_mat4()
        local_position = np.asarray(
            (0.0, -float(screen_height) / 2.0 - float(screen_height) * 0.15
             - float(self._keyboard_height) / 2.0, 0.0),
            dtype=np.float64,
        )
        keyboard_position = (
            screen_pose[:3, 3]
            + screen_pose[:3, :3] @ local_position
            + self._keyboard_position_offset
        )
        # The legacy keyboard is independently head-facing. Do not inherit a
        # room/profile screen rotation when that rotation is not head-facing.
        head = self._head_position_w
        if head is not None:
            direction_from_head = keyboard_position - np.asarray(head, dtype=np.float64)
            distance = float(np.linalg.norm(direction_from_head))
        else:
            direction_from_head = None
            distance = 0.0
        if direction_from_head is not None and distance > 1e-6:
            nx, ny, nz = direction_from_head / distance
            base_yaw = math.atan2(-float(nx), -float(nz))
            base_pitch = math.asin(max(-1.0, min(1.0, float(ny))))
            matrix = euler_to_mat4(
                base_yaw + math.radians(float(self._keyboard_rotation_offset[0])),
                base_pitch + math.radians(float(self._keyboard_rotation_offset[1])),
                0.0,
            ).astype(np.float64)
        else:
            matrix = screen_pose.copy().astype(np.float64)
            local_rotation = euler_to_mat4(
                0.0,
                math.radians(float(self._keyboard_rotation_offset[1])),
                math.radians(float(self._keyboard_rotation_offset[0])),
            ).astype(np.float64)
            matrix[:3, :3] = matrix[:3, :3] @ local_rotation[:3, :3]
        matrix[:3, 3] = keyboard_position
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

    def _controller_interaction_ray(self, hand):
        """Return the legacy-calibrated ray used by the visible laser."""
        aim_matrix = self._aim_mat_l if hand == 0 else self._aim_mat_r
        if aim_matrix is None:
            return None, None
        grip_matrix = self._grip_mat_l if hand == 0 else self._grip_mat_r
        origin, direction = self._get_smoothed_ray(hand)
        has_smoothed_ray = origin is not None and direction is not None
        if not has_smoothed_ray:
            if grip_matrix is not None:
                raw_origin = (
                    grip_matrix[:3, 3] + grip_matrix[:3, 1] * 0.020
                ).astype(np.float64)
            else:
                raw_origin = aim_matrix[:3, 3].astype(np.float64)
            origin = raw_origin
            direction = (-aim_matrix[:3, 2]).astype(np.float64)
        else:
            origin = np.asarray(origin, dtype=np.float64)
            direction = np.asarray(direction, dtype=np.float64)
            if grip_matrix is not None:
                raw_origin = (
                    grip_matrix[:3, 3] + grip_matrix[:3, 1] * 0.020
                ).astype(np.float64)
            else:
                raw_origin = aim_matrix[:3, 3].astype(np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        right_axis = aim_matrix[:3, 0].astype(np.float64)
        right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
        angle = math.radians(12.0)
        direction = self._normalize_interaction_ray(
            direction * math.cos(angle)
            + np.cross(right_axis, direction) * math.sin(angle)
            + right_axis
            * float(np.dot(right_axis, direction))
            * (1.0 - math.cos(angle))
        )
        if has_smoothed_ray:
            # The smoothed ray may leave the finite screen by a small amount
            # while the unsmoothed hand pose is still close to an edge. Copy
            # the legacy edge constraint so the visible laser and interaction
            # hit remain latched to the nearest edge instead of disappearing.
            if self._screen_ray_hit(aim_matrix, raw_origin, direction) is None:
                raw_direction = (-aim_matrix[:3, 2]).astype(np.float64)
                raw_direction /= max(float(np.linalg.norm(raw_direction)), 1e-8)
                raw_direction = self._normalize_interaction_ray(
                    raw_direction * math.cos(angle)
                    + np.cross(right_axis, raw_direction) * math.sin(angle)
                    + right_axis
                    * float(np.dot(right_axis, raw_direction))
                    * (1.0 - math.cos(angle))
                )
                if self._screen_ray_hit(aim_matrix, raw_origin, raw_direction) is None:
                    plane_uv = self._screen_plane_uv(raw_origin, direction)
                    if plane_uv is not None:
                        clamped_u = max(0.0, min(1.0, float(plane_uv[0])))
                        clamped_v = max(0.0, min(1.0, float(plane_uv[1])))
                        clamped_world = self._screen_uv_to_world(clamped_u, clamped_v)
                        edge_direction = clamped_world - raw_origin
                        edge_length = float(np.linalg.norm(edge_direction))
                        if edge_length > 1e-6:
                            edge_direction /= edge_length
                            edge_angle = math.acos(
                                max(-1.0, min(1.0, float(np.dot(raw_direction, edge_direction))))
                            )
                            if edge_angle < self._ray_edge_deadzone_rad:
                                direction = edge_direction
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        return origin, direction

    @staticmethod
    def _normalize_interaction_ray(direction: np.ndarray) -> np.ndarray:
        result = np.asarray(direction, dtype=np.float64)
        result /= max(float(np.linalg.norm(result)), 1e-8)
        return result

    def _screen_plane_uv(self, origin: np.ndarray, direction: np.ndarray):
        """Return unbounded UV on the screen-center plane for edge snapping."""
        if self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation)
        ).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        normal = pose[:3, 2]
        denominator = float(np.dot(normal, direction))
        if abs(denominator) < 1e-6:
            return None
        distance = float(np.dot(normal, pose[:3, 3] - origin) / denominator)
        if distance <= 0.0:
            return None
        hit = np.asarray(origin, dtype=np.float64) + np.asarray(direction, dtype=np.float64) * distance
        local = np.linalg.inv(pose) @ np.append(hit, 1.0)
        return (
            float(local[0]) / max(float(width), 1e-6) + 0.5,
            float(local[1]) / max(float(height), 1e-6) + 0.5,
        )

    def _screen_ray_hit_for_hand(self, hand):
        aim_matrix = self._aim_mat_l if hand == 0 else self._aim_mat_r
        origin, direction = self._controller_interaction_ray(hand)
        if origin is None or direction is None:
            return None
        return self._screen_ray_hit(aim_matrix, origin, direction)

    def _screen_hit_world_for_hand(self, hand):
        """Return the current calibrated laser hit in screen world space."""
        hit = self._screen_ray_hit_for_hand(hand)
        if hit is None or self._filament_screen is None:
            return None
        u, v = (float(hit[0]), float(hit[1]))
        return self._screen_uv_to_world(u, v)

    def _screen_ray_hit(self, matrix, ray_origin=None, ray_direction=None):
        if matrix is None or self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(*(math.radians(float(value)) for value in rotation)).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        origin = (
            matrix[:3, 3].astype(np.float64)
            if ray_origin is None
            else np.asarray(ray_origin, dtype=np.float64)
        )
        direction = (
            (-matrix[:3, 2]).astype(np.float64)
            if ray_direction is None
            else np.asarray(ray_direction, dtype=np.float64)
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-10)
        if self._screen_curved:
            # Match the legacy _laser_screen_hit_uv cylinder geometry and
            # return the same bottom-to-top UV convention used by the GLB
            # screen mesh. Analytic roots avoid a frame-dependent scan.
            half_width = float(width) / 2.0
            half_height = float(height) / 2.0
            half_angle = min(0.72, math.pi / 2.0)
            radius = half_width / max(half_angle, 1e-8)
            local_origin = pose[:3, :3].T @ (origin - pose[:3, 3])
            local_direction = pose[:3, :3].T @ direction
            ox, oy, oz = (float(value) for value in local_origin)
            dx, _dy, dz = (float(value) for value in local_direction)
            qa = dx * dx + dz * dz
            qb = 2.0 * (ox * dx + (oz - radius) * dz)
            qc = ox * ox + (oz - radius) * (oz - radius) - radius * radius
            if abs(qa) < 1e-10:
                return None
            discriminant = qb * qb - 4.0 * qa * qc
            if discriminant < 0.0:
                return None
            root = math.sqrt(max(0.0, discriminant))
            roots = sorted(
                ((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa))
            )
            for distance in roots:
                if distance <= 0.01:
                    continue
                local_hit = local_origin + local_direction * distance
                if abs(float(local_hit[1])) > half_height + 1e-6:
                    continue
                angle = math.atan2(
                    float(local_hit[0]), radius - float(local_hit[2])
                )
                if angle < -half_angle - 1e-6 or angle > half_angle + 1e-6:
                    continue
                return (
                    float((angle + half_angle) / (2.0 * half_angle)),
                    float((float(local_hit[1]) + half_height) / (2.0 * half_height)),
                )
            return None
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
            max(0.0, min(1.0, 0.5 + float(local[1]) / height)),
        )

    def _screen_uv_to_world(self, u: float, v: float) -> np.ndarray | None:
        """Convert screen UV to the current flat or curved screen surface."""
        if self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation)
        ).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        if self._screen_curved:
            half_angle = min(0.72, math.pi / 2.0)
            radius = float(width) / 2.0 / max(half_angle, 1e-8)
            angle = -half_angle + 2.0 * half_angle * float(u)
            local = np.asarray(
                (
                    radius * math.sin(angle),
                    (float(v) - 0.5) * float(height),
                    radius * (1.0 - math.cos(angle)),
                ),
                dtype=np.float64,
            )
        else:
            local = np.asarray(
                (
                    (float(u) - 0.5) * float(width),
                    (float(v) - 0.5) * float(height),
                    0.0,
                ),
                dtype=np.float64,
            )
        return pose[:3, 3] + pose[:3, :3] @ local

    def _set_filament_screen_pose(self, position, rotation=None) -> None:
        if self._filament_screen is None:
            return
        _old_position, width, height, old_rotation = self._filament_screen
        pose_rotation = tuple(rotation if rotation is not None else old_rotation)
        self._filament_screen = (tuple(float(value) for value in position), width, height, pose_rotation)

    def _set_keyboard_world_position(self, position) -> None:
        _screen_position, _width, screen_height, _rotation = self._filament_screen or (
            (0.0, 1.2, -2.0),
            2.4,
            1.35,
            (0.0, 0.0, 0.0),
        )
        # Keep the legacy absolute-position setter compatible with the new
        # screen-relative keyboard anchor.  The caller-provided world
        # position is converted to an offset from the current anchor, so the
        # next pose evaluation reproduces it exactly.
        local_position = np.asarray(
            (
                0.0,
                -float(screen_height) / 2.0
                - float(screen_height) * 0.15
                - float(self._keyboard_height) / 2.0,
                0.0,
            ),
            dtype=np.float64,
        )
        base_position = self._filament_screen_pose_mat4()[:3, 3] + (
            self._filament_screen_pose_mat4()[:3, :3] @ local_position
        )
        self._keyboard_position_offset = (
            np.asarray(position, dtype=np.float64) - base_position
        )

    @staticmethod
    def _rotation_delta_euler_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
        """Convert a relative rotation matrix to the viewer yaw/pitch/roll order."""
        pitch = math.asin(max(-1.0, min(1.0, -float(rotation[1, 2]))))
        cos_pitch = math.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2]))
            roll = math.atan2(float(rotation[1, 0]), float(rotation[1, 1]))
        else:
            yaw = math.atan2(-float(rotation[2, 0]), float(rotation[0, 0]))
            roll = 0.0
        return tuple(math.degrees(value) for value in (yaw, pitch, roll))

    def _apply_grip_screen_rotation(self, hand_index: int) -> None:
        # The legacy right-grip wrist-rotation feature was disabled. Screen
        # rotation remains available only through the legacy left-grip gesture
        # and the documented left-grip/right-stick shortcut.
        if int(hand_index) != 0:
            return
        if self._filament_screen is None:
            return
        suffix = "l" if hand_index == 0 else "r"
        grip_matrix = self._grip_mat_l if hand_index == 0 else self._grip_mat_r
        grip_anchor = getattr(self, f"_grip_rotation_anchor_{suffix}")
        screen_anchor = getattr(self, f"_screen_rotation_anchor_{suffix}")
        if grip_matrix is None or grip_anchor is None or screen_anchor is None:
            return
        relative = (
            np.asarray(grip_matrix[:3, :3], dtype=np.float64)
            @ np.asarray(grip_anchor, dtype=np.float64).T
        )
        _yaw, _pitch, roll = self._rotation_delta_euler_degrees(relative)
        if (
            abs(float(roll)) < 45.0
            or bool(getattr(self, "_grip_screen_rotation_snapped_l", False))
        ):
            return
        direction = 1.0 if float(roll) > 0.0 else -1.0
        rotation = (
            float(screen_anchor[0]),
            max(-89.0, min(89.0, float(screen_anchor[1]))),
            float(screen_anchor[2]) + direction * 90.0,
        )
        self._set_filament_screen_pose(self._filament_screen[0], rotation)
        self._grip_screen_rotation_snapped_l = True

    def _reset_screen_control_hold(self, control: str) -> None:
        setattr(self, f"_screen_{control}_hold_seconds", 0.0)
        setattr(self, f"_screen_{control}_hold_direction", 0)

    def _screen_hold_speed(self, axis_value: float, *, dt: float, control: str) -> float:
        """Return speed from hold duration, restarting after release/reversal."""
        value = float(axis_value)
        if abs(value) <= self._input_deadzone():
            self._reset_screen_control_hold(control)
            return 0.0
        direction = 1 if value > 0.0 else -1
        direction_attr = f"_screen_{control}_hold_direction"
        hold_attr = f"_screen_{control}_hold_seconds"
        if getattr(self, direction_attr) != direction:
            setattr(self, hold_attr, 0.0)
        hold_seconds = min(
            float(self._screen_control_max_hold_seconds),
            float(getattr(self, hold_attr)) + max(0.0, float(dt)),
        )
        setattr(self, direction_attr, direction)
        setattr(self, hold_attr, hold_seconds)
        return min(
            float(self._screen_control_max_speed),
            float(self._screen_control_min_speed)
            + float(self._screen_control_acceleration) * hold_seconds,
        )

    def _apply_right_grip_screen_distance(
        self, joystick_y: float, *, dt: float, laser_hit: Any
    ) -> None:
        """Move the screen radially with five-second hold-time acceleration."""
        if (
            self._filament_screen is None
            or self._head_position_w is None
            or laser_hit is None
            or abs(float(joystick_y)) <= self._input_deadzone()
        ):
            self._reset_screen_control_hold("distance")
            return
        # The Vulkan controller input contract exposes the thumbstick Y axis
        # with the sign flipped from the legacy raw OpenXR value. Restore the
        # legacy sign for this operation: pushing the stick forward must move
        # the screen away from the head.
        legacy_joystick_y = -float(joystick_y)
        speed = self._screen_hold_speed(
            legacy_joystick_y, dt=dt, control="distance"
        )
        if speed <= 0.0:
            return
        position, width, height, rotation = self._filament_screen
        head = np.asarray(self._head_position_w, dtype=np.float64)
        radial = np.asarray(position, dtype=np.float64) - head
        radius = float(np.linalg.norm(radial))
        if radius <= 1e-6:
            return
        radial /= radius
        # Match the legacy sign: positive raw OpenXR Y increases the
        # head-to-screen radius.
        next_radius = max(
            0.3,
            radius + speed * (1.0 if legacy_joystick_y > 0.0 else -1.0) * dt,
        )
        next_position = head + radial * next_radius
        dx, dy, dz = (next_position - head) / next_radius
        next_rotation = (
            math.degrees(math.atan2(-float(dx), -float(dz))),
            math.degrees(math.asin(max(-1.0, min(1.0, float(dy))))),
            float(rotation[2]),
        )
        self._set_filament_screen_pose(next_position, next_rotation)
        self._screen_osd_show_t = time.perf_counter()

    def _apply_right_grip_screen_resize(
        self, joystick_x: float, *, dt: float, laser_hit: Any
    ) -> None:
        """Resize the screen with five-second hold-time acceleration."""
        if (
            self._filament_screen is None
            or laser_hit is None
            or abs(float(joystick_x)) <= self._input_deadzone()
        ):
            self._reset_screen_control_hold("size")
            return
        speed = self._screen_hold_speed(
            float(joystick_x), dt=dt, control="size"
        )
        if speed <= 0.0:
            return
        position, width, height, rotation = self._filament_screen
        next_width = max(0.3, float(width) + math.copysign(speed * dt, float(joystick_x)))
        next_height = next_width * float(height) / max(float(width), 1e-6)
        self._filament_screen = (
            tuple(float(value) for value in position),
            next_width,
            next_height,
            rotation,
        )
        self._screen_osd_show_t = time.perf_counter()

    def _screen_projection_world_points(self) -> np.ndarray | None:
        if self._filament_screen is None:
            return None
        position, width, height, rotation = self._filament_screen
        if width <= 0.0 or height <= 0.0:
            return None
        screen_pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation[:3])
        ).astype(np.float64)
        center = np.asarray(position, dtype=np.float64)
        right = screen_pose[:3, 0].astype(np.float64)
        up = screen_pose[:3, 1].astype(np.float64)
        forward = np.cross(right, up)
        half_width = float(width) * 0.5
        half_height = float(height) * 0.5
        if self._screen_curved:
            segments = 48
            half_angle = 0.72
            radius = half_width / half_angle
            points = []
            for segment in range(segments + 1):
                angle = -half_angle + 2.0 * half_angle * segment / segments
                local_x = radius * math.sin(angle)
                local_z = radius * (1.0 - math.cos(angle))
                column_center = center + right * local_x + forward * local_z
                points.extend((
                    column_center - up * half_height,
                    column_center + up * half_height,
                ))
            return np.asarray(points, dtype=np.float64)
        return np.asarray(
            (
                center - right * half_width - up * half_height,
                center + right * half_width - up * half_height,
                center + right * half_width + up * half_height,
                center - right * half_width + up * half_height,
            ),
            dtype=np.float64,
        )

    def _screen_projection_points(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> np.ndarray | None:
        try:
            sc_w, sc_h = int(swapchain_size[0]), int(swapchain_size[1])
            world_points = self._screen_projection_world_points()
            if sc_w <= 0 or sc_h <= 0 or world_points is None:
                return None
            eye_pose = _xr_view_pose_to_model_mat4(view.pose).astype(np.float64)
            camera_points = (
                np.linalg.inv(eye_pose)
                @ np.concatenate(
                    (world_points, np.ones((len(world_points), 1), dtype=np.float64)),
                    axis=1,
                ).T
            ).T[:, :3]
            depth = -camera_points[:, 2]
            valid = np.isfinite(depth) & (depth > 1e-6)
            if not np.all(valid):
                return None
            fov = view.fov
            tan_left = math.tan(float(fov.angle_left))
            tan_right = math.tan(float(fov.angle_right))
            tan_down = math.tan(float(fov.angle_down))
            tan_up = math.tan(float(fov.angle_up))
            if (
                not all(math.isfinite(value) for value in (
                    tan_left, tan_right, tan_down, tan_up,
                ))
                or tan_right <= tan_left
                or tan_up <= tan_down
            ):
                return None
            ndc_x = 2.0 * (
                camera_points[valid, 0] / depth[valid] - tan_left
            ) / (tan_right - tan_left) - 1.0
            ndc_y = 2.0 * (
                camera_points[valid, 1] / depth[valid] - tan_down
            ) / (tan_up - tan_down) - 1.0
            points = np.column_stack((
                (ndc_x * 0.5 + 0.5) * sc_w,
                (1.0 - (ndc_y * 0.5 + 0.5)) * sc_h,
            ))
            if len(points) != len(world_points) or not np.all(np.isfinite(points)):
                return None
            return points
        except (AttributeError, IndexError, TypeError, ValueError, np.linalg.LinAlgError):
            return None

    def _screen_projection_quad(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self._screen_curved:
            return None
        points = self._screen_projection_points(view, swapchain_size)
        return points if points is not None and points.shape == (4, 2) else None

    def _screen_projection_bounds(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> tuple[float, float, float, float] | None:
        points = self._screen_projection_points(view, swapchain_size)
        if points is None:
            return None
        sc_w, sc_h = int(swapchain_size[0]), int(swapchain_size[1])
        return (
            max(float(np.min(points[:, 0])), 0.0),
            max(float(np.min(points[:, 1])), 0.0),
            min(float(np.max(points[:, 0])), float(sc_w)),
            min(float(np.max(points[:, 1])), float(sc_h)),
        )

    def _screen_footprint_pixels(
        self,
        view: Any,
        swapchain_size: tuple[int, int],
    ) -> tuple[float, float] | None:
        bounds = self._screen_projection_bounds(view, swapchain_size)
        if bounds is None:
            return None
        return max(0.0, bounds[2] - bounds[0]), max(0.0, bounds[3] - bounds[1])

    def _report_screen_resolution(
        self,
        views: list[Any],
        output_frame: VulkanStereoOutputFrame | None,
    ) -> None:
        """Log screen pixel dimensions once per actual resolution configuration."""

        if output_frame is None or self._filament_screen is None:
            return
        sources = (
            (
                int(getattr(output_frame.left_eye, "width", 0)),
                int(getattr(output_frame.left_eye, "height", 0)),
            ),
            (
                int(getattr(output_frame.right_eye, "width", 0)),
                int(getattr(output_frame.right_eye, "height", 0)),
            ),
        )
        targets = self._projection_eye_extents()
        if len(views) < 2 or len(targets) < 2:
            return
        footprints = tuple(
            self._screen_footprint_pixels(views[index], targets[index])
            for index in range(2)
        )
        metadata = dict(output_frame.metadata or {})
        render_size = metadata.get("render_size", metadata.get("source_render_size"))
        if isinstance(render_size, (list, tuple)) and len(render_size) >= 2:
            render_size_label = f"{int(render_size[0])}x{int(render_size[1])}"
        else:
            render_size_label = str(render_size or "unknown")
        screen = self._filament_screen

        # The projected footprint is useful in the message, but it is view-dependent
        # and must not decide whether a resolution diagnostic is emitted.
        resolution_status = (
            sources,
            targets,
            render_size_label,
        )
        if resolution_status == self._last_screen_resolution_status:
            return
        self._last_screen_resolution_status = resolution_status

        def format_size(size: tuple[int, int]) -> str:
            return f"{size[0]}x{size[1]}"

        def format_footprint(footprint: tuple[float, float] | None) -> str:
            if footprint is None:
                return "unknown"
            return f"{round(footprint[0])}x{round(footprint[1])}"

        def format_density(
            source: tuple[int, int], footprint: tuple[float, float] | None
        ) -> str:
            if footprint is None or footprint[0] <= 0.0 or footprint[1] <= 0.0:
                return "unknown"
            return f"{source[0] / footprint[0]:.2f}x{source[1] / footprint[1]:.2f}"

        print(
            "[OpenXRViewer] screen resolution "
            f"source_left={format_size(sources[0])} "
            f"source_right={format_size(sources[1])} "
            f"render_size={render_size_label} "
            f"screen_footprint_left={format_footprint(footprints[0])} "
            f"screen_footprint_right={format_footprint(footprints[1])} "
            f"projection_target_left={format_size(targets[0])} "
            f"projection_target_right={format_size(targets[1])} "
            f"source_per_screen_pixel_left={format_density(sources[0], footprints[0])} "
            f"source_per_screen_pixel_right={format_density(sources[1], footprints[1])} "
            f"screen_m={float(screen[1]):.3f}x{float(screen[2]):.3f} "
            f"distance_m={float(np.linalg.norm(np.asarray(screen[0], dtype=np.float64))):.3f} "
            f"curved={bool(self._screen_curved)}",
            flush=True,
        )

    def _handle_vulkan_pointer_input(self) -> None:
        """Reuse legacy trigger hold/drag semantics for the Vulkan screen."""
        self._right_grip_screen_pointer_applied = False
        now = time.perf_counter()
        inputs = (self._controller_input(0), self._controller_input(1))
        hits = (self._screen_ray_hit_for_hand(0), self._screen_ray_hit_for_hand(1))
        left_grip = bool(inputs[0].get("grip", 0.0) > 0.5)
        right_grip = bool(inputs[1].get("grip", 0.0) > 0.5)
        if not (
            right_grip
            and not left_grip
            and getattr(self, "_grip_target_r", None) == "screen"
            and hits[1] is not None
        ):
            self._reset_screen_control_hold("distance")
            self._reset_screen_control_hold("size")
        stick_active = (
            abs(float(inputs[0].get("joystick_x", 0.0))) > self._input_deadzone()
            or abs(float(inputs[0].get("joystick_y", 0.0))) > self._input_deadzone(),
            abs(float(inputs[1].get("joystick_x", 0.0))) > self._input_deadzone()
            or abs(float(inputs[1].get("joystick_y", 0.0))) > self._input_deadzone(),
        )
        grip_matrices = (self._grip_mat_l, self._grip_mat_r)
        grip_values = (left_grip, right_grip)
        for index, suffix in enumerate(("l", "r")):
            target_attr = f"_grip_target_{suffix}"
            anchor_attr = "_left_grab_anchor" if index == 0 else "_right_grab_anchor"
            rotation_attr = f"_grip_rotation_anchor_{suffix}"
            screen_rotation_attr = f"_screen_rotation_anchor_{suffix}"
            if not grip_values[index]:
                setattr(self, target_attr, None)
                setattr(self, anchor_attr, None)
                setattr(self, rotation_attr, None)
                setattr(self, screen_rotation_attr, None)
                setattr(self, f"_kb_grab_local_{suffix}", None)
                if index == 0:
                    self._grip_screen_rotation_snapped_l = False
                setattr(self, f"_screen_hit_grab_anchor_{suffix}", None)
                continue
            if getattr(self, target_attr) is None:
                # Match the legacy rising-edge latch: a highlighted key is
                # sufficient to select the keyboard, even if the ray is in a
                # key gap on the next frame.
                keyboard_hit = self._keyboard_visible and getattr(
                    self, f"_kb_hover_{suffix}"
                ) is not None
                if keyboard_hit:
                    setattr(self, target_attr, "keyboard")
                elif hits[index] is not None:
                    setattr(self, target_attr, "screen")

            if (
                index == 0
                and getattr(self, target_attr) == "screen"
                and getattr(self, rotation_attr) is None
                and grip_matrices[index] is not None
                and self._filament_screen is not None
            ):
                # The old renderer records the left grip pose on the rising
                # edge. Without this anchor the later 45-degree snap test can
                # never observe wrist rotation.
                setattr(
                    self,
                    rotation_attr,
                    np.asarray(grip_matrices[index][:3, :3], dtype=np.float64).copy(),
                )
                screen_rotation = self._filament_screen[3]
                normalized_roll = ((float(screen_rotation[2]) + 180.0) % 360.0) - 180.0
                if abs(normalized_roll) < 45.0:
                    base_roll = 0.0
                elif 45.0 <= normalized_roll < 135.0:
                    base_roll = 90.0
                elif -135.0 < normalized_roll <= -45.0:
                    base_roll = -90.0
                else:
                    base_roll = 0.0
                setattr(
                    self,
                    screen_rotation_attr,
                    (
                        float(screen_rotation[0]),
                        float(screen_rotation[1]),
                        base_roll,
                    ),
                )
                self._grip_screen_rotation_snapped_l = False

        both_grips = left_grip and right_grip
        if both_grips and not any(stick_active) and all(
            matrix is not None for matrix in grip_matrices
        ):
            common_target = (
                self._grip_target_l
                if self._grip_target_l == self._grip_target_r
                else None
            )
            center = (
                grip_matrices[0][:3, 3].astype(np.float64)
                + grip_matrices[1][:3, 3].astype(np.float64)
            ) * 0.5
            if common_target == "screen" and self._filament_screen is not None:
                if self._both_grip_anchor is None:
                    self._both_grip_anchor = (
                        "screen",
                        np.asarray(self._filament_screen[0], dtype=np.float64) - center,
                    )
                self._set_filament_screen_pose(center + self._both_grip_anchor[1])
            elif common_target == "keyboard":
                if self._both_grip_anchor is None:
                    self._both_grip_anchor = (
                        "keyboard", self._keyboard_pose_mat4()[:3, 3] - center
                    )
                keyboard_position = center + self._both_grip_anchor[1]
                self._set_keyboard_world_position(keyboard_position)
        else:
            self._both_grip_anchor = None
            for index, suffix in enumerate(("l", "r")):
                if not grip_values[index] or grip_matrices[index] is None:
                    continue
                anchor_attr = "_left_grab_anchor" if index == 0 else "_right_grab_anchor"
                rotation_attr = f"_grip_rotation_anchor_{suffix}"
                screen_rotation_attr = f"_screen_rotation_anchor_{suffix}"
                if stick_active[index]:
                    setattr(self, anchor_attr, None)
                    setattr(self, rotation_attr, None)
                    setattr(self, f"_screen_hit_grab_anchor_{suffix}", None)
                    setattr(self, f"_kb_grab_local_{suffix}", None)
                    continue
                target = getattr(self, f"_grip_target_{suffix}")
                if target == "keyboard":
                    # Port the legacy keyboard grip-to-move behavior: keep
                    # the laser's original keyboard-local point attached
                    # while moving the panel on a sphere around the head.
                    ray_origin, ray_direction = self._controller_interaction_ray(index)
                    if ray_origin is None or ray_direction is None:
                        continue
                    keyboard_pose = self._keyboard_pose_mat4()
                    normal = keyboard_pose[:3, 2].astype(np.float64)
                    denominator = float(np.dot(normal, ray_direction))
                    if abs(denominator) < 1e-6:
                        continue
                    distance = float(
                        np.dot(normal, keyboard_pose[:3, 3] - ray_origin)
                        / denominator
                    )
                    if distance < 0.05:
                        continue
                    hit_world = (
                        np.asarray(ray_origin, dtype=np.float64)
                        + np.asarray(ray_direction, dtype=np.float64) * distance
                    )
                    local_hit = np.linalg.inv(keyboard_pose) @ np.append(
                        hit_world, 1.0
                    )
                    keyboard_local_attr = f"_kb_grab_local_{suffix}"
                    keyboard_local = getattr(self, keyboard_local_attr)
                    if keyboard_local is None:
                        setattr(
                            self,
                            keyboard_local_attr,
                            np.asarray(local_hit[:2], dtype=np.float64),
                        )
                        continue

                    desired_center = (
                        hit_world
                        - keyboard_pose[:3, 0] * float(keyboard_local[0])
                        - keyboard_pose[:3, 1] * float(keyboard_local[1])
                    )
                    if self._head_position_w is not None:
                        head = np.asarray(self._head_position_w, dtype=np.float64)
                        current_radius_vector = keyboard_pose[:3, 3] - head
                        current_radius = float(np.linalg.norm(current_radius_vector))
                        desired_radius_vector = desired_center - head
                        desired_radius = float(np.linalg.norm(desired_radius_vector))
                        if current_radius > 1e-6 and desired_radius > 1e-6:
                            desired_center = (
                                head
                                + desired_radius_vector / desired_radius * current_radius
                            )
                    self._set_keyboard_world_position(desired_center)
                elif target == "screen" and self._filament_screen is not None:
                    # Match the legacy renderer: the point selected by the
                    # visible laser stays attached to the same screen-local
                    # coordinate while the hand moves or rotates.
                    hit_world = self._screen_hit_world_for_hand(index)
                    if hit_world is None:
                        continue
                    screen_position, screen_width, screen_height, screen_rotation = (
                        self._filament_screen
                    )
                    screen_pose = self._filament_screen_pose_mat4()
                    screen_center = screen_pose[:3, 3].astype(np.float64)
                    screen_basis = screen_pose[:3, :3].astype(np.float64)
                    hit_anchor_attr = f"_screen_hit_grab_anchor_{suffix}"
                    hit_anchor = getattr(self, hit_anchor_attr)
                    if hit_anchor is None:
                        hit_anchor = screen_basis.T @ (hit_world - screen_center)
                        setattr(self, hit_anchor_attr, hit_anchor)
                    target_center = hit_world - screen_basis @ hit_anchor
                    target_rotation = screen_rotation
                    if index == 1 and self._head_position_w is not None:
                        # Right-hand legacy drag orbits around the head while
                        # preserving the current screen distance and keeps the
                        # screen normal aimed back at the head.
                        head = np.asarray(self._head_position_w, dtype=np.float64)
                        original_radius = float(np.linalg.norm(screen_center - head))
                        radial = target_center - head
                        radial_length = float(np.linalg.norm(radial))
                        if original_radius > 1e-6 and radial_length > 1e-6:
                            target_center = head + radial / radial_length * original_radius
                            dx, dy, dz = (target_center - head) / original_radius
                            target_rotation = (
                                math.degrees(math.atan2(-float(dx), -float(dz))),
                                math.degrees(math.asin(max(-1.0, min(1.0, float(dy))))),
                                float(screen_rotation[2]),
                            )
                    self._set_filament_screen_pose(target_center, target_rotation)
                    if index == 0:
                        self._apply_grip_screen_rotation(index)
        if (
            right_grip
            and not left_grip
            and getattr(self, "_grip_target_r", None) == "screen"
            and self._filament_screen is not None
        ):
            self._right_grip_screen_pointer_applied = True
            input_dt = max(0.001, min(0.1, float(self._last_frame_dt)))
            self._apply_right_grip_screen_resize(
                float(inputs[1].get("joystick_x", 0.0) or 0.0),
                dt=input_dt,
                laser_hit=hits[1],
            )
            self._apply_right_grip_screen_distance(
                float(inputs[1].get("joystick_y", 0.0) or 0.0),
                dt=input_dt,
                laser_hit=hits[1],
            )
        for name, hand, hit, down_flag, up_flag in (
            ("left", inputs[0], hits[0], _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
            ("right", inputs[1], hits[1], _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
        ):
            trigger = float(hand.get("trigger", 0.0) or 0.0)
            state = self._pointer_state[name]
            hand_index = 0 if name == "left" else 1
            aim_matrix = self._aim_mat_l if hand_index == 0 else self._aim_mat_r
            keyboard_hit = False
            ray_origin, ray_direction = self._controller_interaction_ray(hand_index)
            if self._keyboard_visible and aim_matrix is not None:
                keyboard_hit = self._keyboard_plane_hit(
                    ray_origin, ray_direction
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
        self._presenter_thread_id = threading.get_ident()
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
        vulkan_device_lost = bool(
            self.vulkan is not None
            and getattr(self.vulkan, "device_lost", False)
        )
        if self.vulkan is not None and not vulkan_device_lost:
            try:
                self.vulkan.wait_idle()
            except Exception:
                pass

        # Release output-frame leases while their adapters and synchronization
        # objects are still alive.
        self._drop_output_frames()

        if self.vulkan is not None and not vulkan_device_lost:
            try:
                self.vulkan.wait_idle()
            except Exception:
                pass

        # Destroy Filament's external-texture wrappers before the adapters
        # destroy the borrowed screen and Glow VkImages.
        if self.filament_bridge is not None:
            try:
                self.filament_bridge.close()
            except Exception:
                pass
            self.filament_bridge = None

        if self._output_adapter is not None:
            try:
                self._output_adapter.close()
            except Exception:
                pass
            self._output_adapter = None

        if self._vulkan_msdf_quad_renderer is not None:
            try:
                self._vulkan_msdf_quad_renderer.close()
            except Exception:
                pass
            self._vulkan_msdf_quad_renderer = None

        if self._vulkan_projection_screen_pass is not None:
            try:
                self._vulkan_projection_screen_pass.close()
            except Exception:
                pass
            self._vulkan_projection_screen_pass = None

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
            self._multiview_active = False

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
            if not vulkan_device_lost:
                try:
                    xr.destroy_instance(self.instance)
                except Exception:
                    pass
            self.instance = None

        self.system_id = None
        self.swapchain_format = None
        self._tool_quad_swapchain_format = None
        self._graphics_binding = None
        self._initialized = False
        self._last_screen_resolution_status = None
        self._last_screen_resolution_log_t = 0.0
        self._clear_presenter_commands()
        self._drop_output_frames()
        self._has_presented_frame = False
        self._last_quad_layers = []
        self._last_screen_quad_layers = []
        for host_image in tuple(
            self._visual_regression_source_host_images.values()
        ) + tuple(self._visual_regression_projection_host_images.values()):
            try:
                host_image.close()
            except Exception:
                pass
        self._visual_regression_source_host_images.clear()
        self._visual_regression_projection_host_images.clear()
        self._visual_regression_capture_eyes.clear()
        self._visual_regression_capture_failed = False
        self._source_frame_wait_logged = False
        self._accept_output = False
        self._filament_animation_origin = None
        self._profile_initial_head = None
        self._profile_space_applied = False
        self._profile_alignment_logged = False
        self._reference_space_type = None
        self._presenter_thread_id = None
        self._next_output_frame_id = 0

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
                vk, vk_physical_device, require_multiview=True
            )
        except VulkanCapabilityError as exc:
            raise OpenXrVulkanUnavailableError(str(exc)) from exc
        queue_family_properties = vk.vkGetPhysicalDeviceQueueFamilyProperties(
            vk_physical_device
        )
        available_queue_count = int(
            queue_family_properties[queue_family_index].queueCount
        )
        requested_queue_count = 2 if available_queue_count >= 2 else 1
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=queue_family_index,
            queueCount=requested_queue_count,
            pQueuePriorities=[1.0, 0.5][:requested_queue_count],
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
            compute_queue_index=1 if requested_queue_count >= 2 else 0,
        )
        print(
            "[OpenXRViewer] Vulkan queue topology: "
            f"graphics=family{queue_family_index}/queue0 "
            f"glow_compute=family{queue_family_index}/queue"
            f"{1 if requested_queue_count >= 2 else 0} "
            f"async={requested_queue_count >= 2}",
            flush=True,
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
            self.swapchains.append(self._create_projection_swapchain(width, height))

    def _create_projection_swapchain(
        self, width: int, height: int, *, array_size: int = 1
    ) -> _EyeSwapchain:
        xr = self.xr
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
                array_size=array_size,
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
        return _EyeSwapchain(
            handle=handle,
            images=images,
            width=width,
            height=height,
            resources=self._register_swapchain_images(images, width, height),
            array_size=array_size,
        )

    def _destroy_projection_swapchain(self, eye: _EyeSwapchain) -> None:
        for resource in reversed(eye.resources):
            self.vulkan.unregister_external_image(resource)
        self.xr.destroy_swapchain(eye.handle)

    def _projection_eye_extents(self) -> tuple[tuple[int, int], ...]:
        if len(self.swapchains) == 1 and self.swapchains[0].array_size >= 2:
            extent = (self.swapchains[0].width, self.swapchains[0].height)
            return (extent, extent)
        return tuple((eye.width, eye.height) for eye in self.swapchains[:2])

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
        if (
            self._presenter_thread_id is not None
            and threading.get_ident() != self._presenter_thread_id
        ):
            self._enqueue_presenter_command("submit_output", frame)
            return
        self._submit_output_on_presenter(frame)

    def submit_runtime_result(self, runtime_result: Any, timestamp: float) -> None:
        """Marshal raw inference output to the Presenter-owned Vulkan path."""

        if not self._accept_output or not self.session_running:
            return
        payload = (runtime_result, float(timestamp))
        if (
            self._presenter_thread_id is not None
            and threading.get_ident() != self._presenter_thread_id
        ):
            self._enqueue_presenter_command("submit_runtime_result", payload)
            return
        self._submit_runtime_result_on_presenter(*payload)

    def _submit_runtime_result_on_presenter(
        self, runtime_result: Any, timestamp: float
    ) -> None:
        """Convert and publish inference output while owning the Vulkan context."""

        debug_info = dict(getattr(runtime_result, "debug_info", None) or {})
        requested_backend = (
            "vulkan_zero_copy"
            if getattr(runtime_result, "vulkan_compute_request", None) is not None
            else (
                "vulkan_host"
                if str(debug_info.get("stereo_compute_backend", "")).strip().lower()
                == "vulkan"
                else None
            )
        )
        if (
            self._output_adapter is not None
            and requested_backend is not None
            and getattr(self._output_adapter, "backend_name", None) != requested_backend
        ):
            close = getattr(self._output_adapter, "close", None)
            if callable(close):
                close()
            self._output_adapter = None
        if self._output_adapter is None:
            from app_runtime.gpu_producer import create_gpu_producer_adapter

            try:
                self._output_adapter = create_gpu_producer_adapter(
                    self,
                    backend=requested_backend,
                )
                self._output_adapter_error = None
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != self._output_adapter_error:
                    print(
                        f"[OpenXRViewer] GPU producer adapter unavailable: {message}; "
                        "waiting for a compatible GPU interop adapter",
                        flush=True,
                    )
                    self._output_adapter_error = message
                return
        try:
            conversion_started = time.perf_counter()
            left_eye = getattr(runtime_result, "left_eye", None)
            right_eye = getattr(runtime_result, "right_eye", None)
            if not isinstance(left_eye, VulkanImageResource) or not isinstance(
                right_eye, VulkanImageResource
            ):
                frame = self._output_adapter.convert(
                    runtime_result,
                    frame_id=self._next_output_frame_id,
                    timestamp=timestamp,
                )
            else:
                debug_info = dict(getattr(runtime_result, "debug_info", None) or {})
                frame = VulkanStereoOutputFrame(
                    frame_id=self._next_output_frame_id,
                    timestamp=timestamp,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    sbs=getattr(runtime_result, "sbs", None),
                    ready_timeline=getattr(runtime_result, "ready_timeline", None),
                    metadata=debug_info,
                    color_space=str(debug_info.get("output_color_space", "srgb")),
                    image_origin=str(debug_info.get("output_image_origin", "top_left")),
                )
            callback = self._on_breakdown_add_time
            if callback is not None:
                callback(
                    "openxr_vulkan_output_convert",
                    max(0.0, time.perf_counter() - conversion_started),
                )
                metadata = dict(getattr(frame, "metadata", None) or {})
                callback(
                    "openxr_vulkan_input_slot_wait",
                    max(
                        0.0,
                        float(metadata.get("vulkan_input_slot_wait_ms", 0.0))
                        / 1000.0,
                    ),
                )
                callback(
                    "openxr_vulkan_input_upload",
                    max(
                        0.0,
                        float(metadata.get("vulkan_input_upload_ms", 0.0))
                        / 1000.0,
                    ),
                )
            self._next_output_frame_id += 1
        except Exception as exc:
            print(
                f"[OpenXRViewer] Runtime output conversion failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        self._submit_output_on_presenter(frame)

    def _submit_output_on_presenter(self, frame: VulkanStereoOutputFrame) -> None:
        with self._output_lock:
            previous = self._pending_output
            self._pending_output = frame
        if previous is not None and previous is not frame:
            self._release_output_frame(previous)

    def _enqueue_presenter_command(self, kind: str, payload: Any) -> None:
        command = (str(kind), payload)
        while True:
            try:
                self._presenter_commands.put_nowait(command)
                return
            except queue.Full:
                try:
                    old_kind, old_payload = self._presenter_commands.get_nowait()
                except queue.Empty:
                    continue
                if old_kind == "submit_output":
                    self._release_output_frame(old_payload)

    def _drain_presenter_commands(self) -> None:
        latest_runtime_result = None
        while True:
            try:
                kind, payload = self._presenter_commands.get_nowait()
            except queue.Empty:
                break
            if kind == "submit_output":
                self._submit_output_on_presenter(payload)
            elif kind == "submit_runtime_result":
                # Convert at most one raw result per XR tick. Processing a
                # burst here can exhaust the output ring before the previous
                # frame reaches commit/release, causing the owner thread to
                # wait on its own slot lease indefinitely.
                latest_runtime_result = payload
        if latest_runtime_result is not None:
            self._submit_runtime_result_on_presenter(*latest_runtime_result)

    def _clear_presenter_commands(self) -> None:
        while True:
            try:
                kind, payload = self._presenter_commands.get_nowait()
            except queue.Empty:
                return
            if kind == "submit_output":
                self._release_output_frame(payload)
            elif kind == "submit_runtime_result":
                continue

    @staticmethod
    def _release_output_frame(frame: VulkanStereoOutputFrame | None) -> None:
        if frame is None:
            return
        metadata = frame.metadata or {}
        if metadata.get("_vulkan_release_attempted"):
            return
        # A frame can appear in displayed/rendering/pending bookkeeping while
        # unwinding an exception.  Its callbacks own idempotent CPU cleanup,
        # but Vulkan submissions must never be attempted twice—especially
        # after VK_ERROR_DEVICE_LOST made every handle terminal.
        metadata["_vulkan_release_attempted"] = True
        consumer_release = metadata.get("_vulkan_source_consumer_release")
        consumer_semaphores = metadata.get(
            "_vulkan_consumer_release_semaphores"
        )
        consumer_timeline = metadata.get("_vulkan_consumer_release_timeline")
        try:
            if callable(consumer_release) and consumer_timeline is not None:
                consumer_release(
                    frame.frame_id,
                    wait_for_timeline=int(consumer_timeline),
                )
            elif callable(consumer_release) and consumer_semaphores is not None:
                consumer_release(frame.frame_id, tuple(consumer_semaphores))
            else:
                callback = metadata.get("_vulkan_output_release")
                if callable(callback):
                    fallback_timeline = metadata.get("_vulkan_fallback_copy_timeline")
                    if fallback_timeline is not None:
                        try:
                            callback(
                                frame.frame_id,
                                wait_for_timeline=int(fallback_timeline),
                            )
                        except TypeError:
                            callback(frame.frame_id)
                    else:
                        callback(frame.frame_id)
        finally:
            glow_release = metadata.get("_vulkan_glow_release")
            if callable(glow_release):
                glow_release(frame.frame_id)

    def release_displayed_output_for_reuse(self, slot_index: int) -> bool:
        """Release a displayed source slot before a producer ring wrap blocks."""

        with self._output_lock:
            displayed = self._displayed_output
            if displayed is None:
                return False
            metadata = displayed.metadata or {}
            if metadata.get("vulkan_output_ring_slot") != int(slot_index):
                return False
            self._displayed_output = None
        self._release_output_frame(displayed)
        return True

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
            release_started = time.perf_counter()
            self._release_output_frame(previous)
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_output_release",
                    time.perf_counter() - release_started,
                )

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
        file_reader, asset_reads = self._start_filament_file_reads()
        try:
            bridge.create(
                instance=self.vulkan.instance,
                physical_device=self.vulkan.physical_device,
                device=self.vulkan.device,
                queue_family_index=self.vulkan.queue_family_index,
                queue_index=0,
            )
            self._multiview_active = self._try_enable_filament_multiview(bridge)
            if not self._multiview_active:
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
                bridge.load_glb(asset_reads["environment"].result())
            if (
                self._controller_brand is not None
                and getattr(bridge, "controller_abi_available", True)
                and hasattr(bridge, "load_controller")
            ):
                bridge.load_controller(0, asset_reads["controller_left"].result())
                bridge.load_controller(1, asset_reads["controller_right"].result())
                print(
                    "Filament controllers loaded: "
                    f"brand={self._controller_brand.name} "
                    f"abi={bridge.controller_abi_available} "
                    f"visibility_abi={getattr(bridge, 'controller_visibility_abi_available', False)} "
                    f"laser_abi={getattr(bridge, 'laser_abi_available', False)}",
                    flush=True,
                )
            if (
                getattr(bridge, "controller_guide_abi_available", False)
                and hasattr(bridge, "set_controller_guide_texture")
            ):
                if self._controller_callout_rgba is None:
                    self._controller_callout_rgba = build_controller_callout_rgba(lang="CN")
                bridge.set_controller_guide_texture(self._controller_callout_rgba)
                print(
                    "Filament controller guide loaded: projection_layer=True",
                    flush=True,
                )
            bridge.set_scene_exposure(self._filament_scene_exposure)
            bridge.set_skybox_brightness(self._filament_skybox_brightness)
            if hasattr(bridge, "set_ambient_light"):
                bridge.set_ambient_light(self._controller_ambient_light_color())
            if hasattr(bridge, "set_controller_ambient_light"):
                bridge.set_controller_ambient_light(
                    self._controller_hdr_ambient_light_color(),
                    True,
                )
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
        finally:
            file_reader.shutdown(wait=True)

    def _report_projection_composer_boundary(self) -> None:
        print(
            "[OpenXRViewer] Vulkan projection composer boundary: "
            f"requested={self._vulkan_projection_composer_requested} "
            "active=False fallback=existing_projection_path",
            flush=True,
        )

    def _projection_screen_push_constants(
        self, view: Any, sampling_constants: bytes | None = None
    ) -> bytes:
        if self._filament_screen is None:
            raise RuntimeError("Vulkan Projection Composer screen is unavailable")
        position, width, height, rotation = self._filament_screen
        screen_rotation = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation[:3])
        ).astype(np.float32)
        view_projection = (
            _fov_to_proj_mat4_d3d(
                view.fov,
                near=self._profile_near_plane,
                far=self._profile_far_plane,
            )
            @ _pose_to_view_mat4(view.pose)
        ).astype(np.float32)
        # Vulkan's positive-height viewport maps positive NDC Y downward.
        view_projection[1, :] *= -1.0
        if sampling_constants is None:
            sampling_values = np.zeros(4, dtype=np.float32)
        else:
            sampling_values = np.frombuffer(sampling_constants, dtype="<f4")
            if sampling_values.size != 4 or not np.all(np.isfinite(sampling_values)):
                raise ValueError("Vulkan Projection Composer sampling constants are invalid")
        values = np.concatenate((
            view_projection.reshape(-1, order="F"),
            np.asarray((*position, sampling_values[0]), dtype=np.float32),
            np.asarray((*screen_rotation[:3, 0], sampling_values[1]), dtype=np.float32),
            np.asarray((*screen_rotation[:3, 1], sampling_values[2]), dtype=np.float32),
            np.asarray((
                float(width) * 0.5,
                float(height) * 0.5,
                0.72 if self._screen_curved else 0.0,
                sampling_values[3],
            ), dtype=np.float32),
        )).astype("<f4", copy=False)
        if values.size != 32 or not np.all(np.isfinite(values)):
            raise ValueError("Vulkan Projection Composer screen transform is invalid")
        return values.tobytes()

    def _projection_screen_sampling_constants(self, source: Any, target: Any) -> bytes:
        width = int(getattr(source, "width", 0))
        height = int(getattr(source, "height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("Vulkan Projection Composer source size is invalid")
        target_width = int(getattr(target, "width", 0))
        target_height = int(getattr(target, "height", 0))
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Vulkan Projection Composer target size is invalid")
        # The quality chain runs before world-space projection. The final
        # screen pass always samples the completed quality mip texture.
        values = np.asarray(
            (
                1.0 / float(width),
                1.0 / float(height),
                1.0,
                0.0,
            ),
            dtype="<f4",
        )
        return values.tobytes()

    def _apply_vulkan_projection_sampling(
        self,
        frame: VulkanStereoOutputFrame,
        *,
        quality_chain_enabled: bool = True,
    ) -> None:
        screen_pass = self._vulkan_projection_screen_pass
        if screen_pass is None:
            return
        metadata = frame.metadata or {}
        try:
            if not quality_chain_enabled:
                screen_pass.set_sampling_config(
                    min_lod=0.0,
                    max_lod=0.0,
                    mip_lod_bias=0.0,
                    rcas_sharpness=0.0,
                )
                return
            screen_pass.set_sampling_config(
                min_lod=metadata.get("vulkan_projection_min_lod", 0.0),
                max_lod=metadata.get("vulkan_projection_max_lod", 0.35),
                mip_lod_bias=metadata.get("vulkan_projection_mip_lod_bias", -0.35),
                rcas_sharpness=metadata.get("vulkan_projection_rcas_sharpness", 0.5),
            )
        except (TypeError, ValueError):
            return

    def _render_vulkan_projection_composer(
        self,
        frame: VulkanStereoOutputFrame,
        acquired_images: list[tuple[_EyeSwapchain, int]],
        views: list[Any],
    ) -> int:
        if self.vulkan is None or len(acquired_images) not in {1, 2}:
            raise RuntimeError("Vulkan Projection Composer has no valid targets")
        diagnostic = _env_flag("D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC")
        layered = len(acquired_images) == 1 and acquired_images[0][0].array_size >= 2
        if diagnostic:
            target_eye, image_index = acquired_images[0]
            colors = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))
            timelines = []
            for eye_index, color in enumerate(colors):
                eye_target, target_index = (
                    (target_eye, image_index)
                    if layered
                    else acquired_images[eye_index]
                )
                timelines.append(
                    self.vulkan.clear_color_image(
                        eye_target.resources[target_index].image,
                        color,
                        base_array_layer=eye_index if layered else 0,
                    )
                )
            print(
                "[OpenXRViewer] Vulkan projection composer eye diagnostic: "
                f"left=red layer={0 if layered else 'eye0'} "
                f"right=green layer={1 if layered else 'eye1'}",
                flush=True,
            )
            self._vulkan_projection_composer_frame_id = int(frame.frame_id)
            self._vulkan_projection_composer_active = True
            return max(timelines)
        if len(views) < 2:
            raise RuntimeError("Vulkan Projection Composer requires two views")
        prepare_source = (frame.metadata or {}).get(
            "_vulkan_source_prepare_for_sampling"
        )
        if not callable(prepare_source):
            raise RuntimeError(
                "Vulkan Projection Composer source preparation is unavailable"
            )
        source_inputs = (frame.left_eye, frame.right_eye)
        status = (
            layered,
            int(source_inputs[0].width),
            int(source_inputs[0].height),
            int(acquired_images[0][0].width),
            int(acquired_images[0][0].height),
            bool(self._screen_curved),
        )
        if status != self._last_vulkan_projection_composer_status:
            self._last_vulkan_projection_composer_status = status
            print(
                "[OpenXRViewer] Vulkan projection composer active: "
                f"mode=graphics_triangle_strip layered={layered} "
                f"source={status[1]}x{status[2]} "
                f"target={status[3]}x{status[4]} curved={status[5]}",
                flush=True,
            )
        target_format = int(acquired_images[0][0].resources[0].format)
        if self._vulkan_projection_screen_pass is None:
            self._vulkan_projection_screen_pass = VulkanProjectionScreenPass(
                self.vulkan, target_format
            )
        self._apply_vulkan_projection_sampling(
            frame,
            quality_chain_enabled=self._vulkan_projection_quality_chain_requested,
        )
        plan = self._active_screen_sampling_plan
        use_quality_mip = bool(
            self._vulkan_projection_quality_chain_requested and plan is not None
        )
        projection_draws = []
        glow_source = (frame.metadata or {}).get("glow_vulkan_image")
        for eye_index, source in enumerate(source_inputs):
            source_prepare_started = time.perf_counter()
            wait_semaphore = prepare_source(frame.frame_id, eye_index)
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_vulkan_composer_source_prepare",
                    time.perf_counter() - source_prepare_started,
                )
            draw_prepare_started = time.perf_counter()
            target_eye, image_index = (
                acquired_images[0] if layered else acquired_images[eye_index]
            )
            sampling_constants = self._projection_screen_sampling_constants(
                source, target_eye.resources[image_index]
            )
            projection_draws.append(
                {
                    "source": source,
                    "target": target_eye.resources[image_index],
                    "array_layer": eye_index if layered else 0,
                    "eye_index": eye_index,
                    "frame_slot": int(self.frame_count) % 3,
                    "push_constants": self._projection_screen_push_constants(
                        views[eye_index], sampling_constants
                    ),
                    "clear_color": self.config.clear_color,
                    "wait_semaphore": wait_semaphore,
                    "glow_source": glow_source,
                }
            )
            if self._on_breakdown_add_time is not None:
                self._on_breakdown_add_time(
                    "openxr_vulkan_composer_draw_prepare",
                    time.perf_counter() - draw_prepare_started,
                )
        submit_started = time.perf_counter()
        timeline = None
        if use_quality_mip:
            try:
                timeline = self._vulkan_projection_screen_pass.try_submit_stereo_quality_mip(
                    projection_draws,
                    mode=plan.mode,
                    filter_scale=plan.filter_scale,
                    upscale_scale=plan.upscale_scale,
                )
            except Exception as exc:
                print(
                    "[OpenXRViewer] Vulkan projection quality chain skipped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if self._on_breakdown_inc is not None and use_quality_mip:
            if timeline is not None:
                self._on_breakdown_inc("openxr_vulkan_composer_quality", 1)
            else:
                self._on_breakdown_inc("openxr_vulkan_composer_quality_skip", 1)
        if timeline is None:
            timeline = self._vulkan_projection_screen_pass.submit_stereo(projection_draws)
        if glow_source is not None and self._filament_glow_environment_enabled:
            timeline = self._vulkan_projection_screen_pass.submit_stereo_glow(
                projection_draws, wait_for_timeline=int(timeline)
            )
            if self._on_breakdown_inc is not None:
                self._on_breakdown_inc("openxr_vulkan_composer_glow", 1)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_vulkan_composer_submit",
                time.perf_counter() - submit_started,
            )
            submit_profile = self._vulkan_projection_screen_pass.last_submit_profile
            for stage, metric in (
                ("fence_wait", "openxr_vulkan_composer_fence_wait"),
                ("record", "openxr_vulkan_composer_record"),
                ("queue_submit", "openxr_vulkan_composer_queue_submit"),
            ):
                if stage in submit_profile:
                    self._on_breakdown_add_time(metric, submit_profile[stage])
        self._vulkan_projection_composer_frame_id = int(frame.frame_id)
        self._vulkan_projection_composer_active = True
        return int(timeline)

    def _try_enable_filament_multiview(self, bridge: Any) -> bool:
        if not (
            getattr(bridge, "multiview_abi_available", False)
            and getattr(bridge, "multiview_supported", False)
            and len(self.swapchains) == 2
        ):
            return False
        left, right = self.swapchains
        if (left.width, left.height) != (right.width, right.height):
            print(
                "[OpenXRViewer] Filament multiview unavailable: eye extents differ",
                flush=True,
            )
            return False
        layered = self._create_projection_swapchain(
            left.width, left.height, array_size=2
        )
        try:
            bridge.create_stereo_swapchain(
                (image.image for image in layered.images),
                format=self.swapchain_format,
                width=layered.width,
                height=layered.height,
            )
        except Exception as exc:
            self._destroy_projection_swapchain(layered)
            print(
                "[OpenXRViewer] Filament multiview fallback: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        previous = self.swapchains
        self.swapchains = [layered]
        for eye in previous:
            self._destroy_projection_swapchain(eye)
        print(
            "[OpenXRViewer] Filament projection path: "
            f"multiview array_size=2 extent={layered.width}x{layered.height}",
            flush=True,
        )
        return True

    def _initialize_msdf_text_atlas(self) -> None:
        """Load the shared atlas for the MSDF-to-Quad OSD path."""
        try:
            atlas = MsdfFontAtlas()
        except Exception as exc:
            self._msdf_font_atlas = None
            print(
                "[OpenXRViewer] MSDF atlas unavailable; "
                f"using legacy Quad text ({type(exc).__name__}: {exc})",
                flush=True,
            )
            return
        self._msdf_font_atlas = atlas
        print(
            "[OpenXRViewer] MSDF atlas loaded for Quad OSD: "
            f"pages={len(atlas.pages)} glyphs={len(atlas.glyphs)} "
            f"distance_range={atlas.distance_range:g}",
            flush=True,
        )

    def _apply_screen_sampling_policy(
        self,
        output_frame: VulkanStereoOutputFrame | None,
    ) -> ScreenSamplingPlan | None:
        """Apply the GUI-headset/input-resolution matrix to the screen filter."""
        if output_frame is None or self._filament_screen is None:
            return None
        metadata = dict(output_frame.metadata or {})

        def metadata_size(value: Any) -> tuple[int, int] | None:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    width, height = int(value[0]), int(value[1])
                except (TypeError, ValueError):
                    return None
                return (width, height) if width > 0 and height > 0 else None
            text = str(value or "").strip().lower()
            if "x" not in text:
                return None
            left, right = text.split("x", 1)
            try:
                width, height = int(left), int(right)
            except ValueError:
                return None
            return (width, height) if width > 0 and height > 0 else None

        # capture_size is attached by the capture pipeline and represents the
        # GUI input screen. The eye image is only the fallback after processing.
        source_size = next(
            (
                metadata_size(metadata.get(key))
                for key in ("capture_size", "source_size", "input_size")
                if metadata_size(metadata.get(key)) is not None
            ),
            None,
        )
        if source_size is None:
            source = output_frame.left_eye
            source_size = (
                int(getattr(source, "width", 0)),
                int(getattr(source, "height", 0)),
            )
        try:
            plan = build_screen_sampling_plan(
                source_size[0],
                source_size[1],
                self._headset_preset.resolution_tier_k,
            )
        except (TypeError, ValueError):
            return None
        status = (
            plan.source_width,
            plan.source_height,
            plan.input_tier_k,
            plan.headset_tier_k,
            plan.recommended_headset_tier_k,
            plan.effective_tier_k,
            round(plan.filter_scale, 4),
            round(plan.upscale_scale, 4),
            plan.mode,
        )
        status_changed = status != self._last_screen_sampling_status
        if status_changed:
            self._last_screen_sampling_status = status
            print(
                "[OpenXRViewer] screen sampling policy "
                f"headset={self._headset_preset.key} "
                f"headset_tier={plan.headset_tier_k}K "
                f"input={plan.source_width}x{plan.source_height} "
                f"input_tier={plan.input_tier_k}K "
                f"recommended={plan.recommended_headset_tier_k}K "
                f"effective={plan.effective_tier_k}K "
                f"filter_scale={plan.filter_scale:.2f} mode={plan.mode} "
                f"upscale_scale={plan.upscale_scale:.2f} "
                "sampling_owner="
                + (
                    "vulkan_projection_composer"
                ),
                flush=True,
            )
        self._active_screen_sampling_plan = plan
        return plan

    def _initialize_msdf_quad_renderer(self) -> None:
        if self.vulkan is None or self._msdf_font_atlas is None:
            return
        try:
            self._vulkan_msdf_quad_renderer = VulkanMsdfQuadRenderer(
                self.vulkan, self._msdf_font_atlas
            )
            print(
                "[OpenXRViewer] Vulkan MSDF Quad renderer active: "
                "atlas_gpu=True output=storage_image",
                flush=True,
            )
        except Exception as exc:
            self._vulkan_msdf_quad_renderer = None
            print(
                "[OpenXRViewer] Vulkan MSDF Quad renderer unavailable; "
                f"using CPU MSDF compatibility path ({type(exc).__name__}: {exc})",
                flush=True,
            )

    def _submit_msdf_text_runs(self, runs: list[dict[str, Any]]) -> bool:
        """Submit merged MSDF runs; return false when the legacy path is needed."""
        bridge = self.filament_bridge
        atlas = self._msdf_font_atlas
        if bridge is None or atlas is None or not getattr(
            bridge, "text_overlay_abi_available", False
        ):
            return False
        grouped: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        try:
            for run in runs:
                geometry = atlas.build_geometry(**run)
                for page, buffers in geometry.items():
                    grouped.setdefault(page, []).append(buffers)
            for page in range(len(atlas.pages)):
                buffers = grouped.get(page, ())
                if buffers:
                    vertices = np.ascontiguousarray(
                        np.concatenate([item[0] for item in buffers], axis=0),
                        dtype=np.float32,
                    )
                    index_parts = []
                    vertex_offset = 0
                    for item_vertices, item_indices in buffers:
                        index_parts.append(item_indices.astype(np.uint32) + vertex_offset)
                        vertex_offset += int(item_vertices.shape[0])
                    indices = np.ascontiguousarray(
                        np.concatenate(index_parts).astype(np.uint16), dtype=np.uint16
                    )
                    bridge.set_text_overlay_page(page, vertices, indices, visible=True)
                else:
                    bridge.set_text_overlay_page(
                        page,
                        np.zeros((0, 9), dtype=np.float32),
                        np.zeros(0, dtype=np.uint16),
                        visible=False,
                    )
        except Exception as exc:
            self._msdf_font_atlas = None
            print(
                "[OpenXRViewer] MSDF text disabled after submit failure; "
                "retaining legacy overlay: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        return True

    def _start_filament_file_reads(
        self,
    ) -> tuple[ThreadPoolExecutor, dict[str, Future[bytes]]]:
        """Read assets off-thread without moving any Filament work off-owner."""
        executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="FilamentFileIO")
        reads: dict[str, Future[bytes]] = {}
        if self.config.filament_glb_path:
            reads["environment"] = executor.submit(
                Path(self.config.filament_glb_path).read_bytes
            )
        if self._controller_brand is not None:
            reads["controller_left"] = executor.submit(
                self._controller_brand.left_glb.read_bytes
            )
            reads["controller_right"] = executor.submit(
                self._controller_brand.right_glb.read_bytes
            )
        return executor, reads

    def _update_filament_controllers(self, bridge: Any) -> None:
        self._update_filament_controller_guide(bridge)
        if (
            self._controller_brand is None
            or not getattr(bridge, "controller_abi_available", True)
            or not hasattr(bridge, "set_controller_pose")
            or not hasattr(bridge, "set_controller_inputs")
        ):
            return
        offset = np.eye(4, dtype=np.float32)
        offset[:3, 3] = np.asarray(
            self._controller_calibration_offset, dtype=np.float32
        )
        # Controller profiles use the legacy model calibration convention:
        # model_rotation_deg is a rotation around the local X axis.
        rotation = euler_to_mat4(
            0.0, math.radians(self._controller_calibration_rotation_deg), 0.0
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
            if max(
                values.get("joystick_touched", 0.0),
                values.get("touchpad_touched", 0.0),
            ) > 0.5:
                # Keep the frozen C ABI: bit 6 carries the shared WebXR touch state.
                button_mask |= 1 << 6
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
                    smoothed_origin, direction = self._controller_interaction_ray(hand)
                    if smoothed_origin is None or direction is None:
                        bridge.set_controller_laser(
                            hand, np.eye(4, dtype=np.float32), visible=False
                        )
                        continue
                    right_axis = aim_matrix[:3, 0].astype(np.float64)
                    right_axis /= max(float(np.linalg.norm(right_axis)), 1e-8)
                    # Start the beam just beyond the grip shell.
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
    def _update_filament_controller_guide(self, bridge: Any) -> None:
        if (
            getattr(bridge, "controller_guide_abi_available", False)
            and hasattr(bridge, "set_controller_guide")
        ):
            geometry = self._controller_guide_geometry()
            if geometry is None:
                bridge.set_controller_guide(np.eye(4, dtype=np.float32), visible=False)
            else:
                position, size, basis = geometry
                guide_matrix = np.eye(4, dtype=np.float32)
                guide_matrix[:3, 0] = (basis[:, 0] * size[0]).astype(np.float32)
                guide_matrix[:3, 1] = (basis[:, 1] * size[1]).astype(np.float32)
                guide_matrix[:3, 2] = basis[:, 2].astype(np.float32)
                guide_matrix[:3, 3] = np.asarray(position, dtype=np.float32)
                bridge.set_controller_guide(guide_matrix, visible=True)

    def _load_filament_profile(self) -> None:
        profile_path = self.config.filament_profile_path
        if not profile_path:
            return
        with open(profile_path, "r", encoding="utf-8-sig") as handle:
            profile = json.load(handle)
        if not isinstance(profile, dict):
            raise ValueError("Filament profile root must be an object")

        presets = profile.get("lighting_presets")
        self._filament_lighting_presets = tuple(
            item for item in presets if isinstance(item, dict)
        ) if isinstance(presets, list) else ()
        try:
            self._filament_lighting_preset_index = int(
                profile.get("lighting_preset_index", 0)
            )
        except (TypeError, ValueError):
            self._filament_lighting_preset_index = 0
        if self._filament_lighting_presets:
            self._filament_lighting_preset_index %= len(
                self._filament_lighting_presets
            )
        self._apply_filament_glow_profile_fields(profile)

        view_pose = profile.get("view_pose", profile.get("camera"))
        view_poses = profile.get("view_poses")
        if isinstance(view_poses, list) and view_poses:
            index = int(profile.get("view_pose_index", 0)) % len(view_poses)
            view_pose = view_poses[index]
        if not isinstance(view_pose, dict):
            # Default and panorama environments intentionally have no authored
            # room-space seat. Rebase identity to the initial leveled headset
            # pose, matching the legacy OpenXR default screen contract.
            view_pose = {
                "name": "Default",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "rotation_deg": [0.0, 0.0, 0.0],
            }

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

            pose_space = str(
                view_pose.get(
                    "view_pose_space",
                    view_pose.get("pose_space", profile.get("view_pose_space", "world")),
                )
            ).strip().lower()
            if pose_space in {"scene", "glb", "local"}:
                glb_position = world_position_vec
            else:
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
        ambient_color = profile.get(
            "env_ambient_color", self._filament_ambient_light_color
        )
        if isinstance(ambient_color, (list, tuple)) and len(ambient_color) >= 3:
            self._filament_ambient_light_color = tuple(
                max(0.0, float(value)) for value in ambient_color[:3]
            )
        # Match the legacy controller renderer: a unit-less head light follows
        # the eye, while the Filament bridge supplies the fixed top fill.
        fill_color = profile.get("env_head_light_color", self._filament_fill_light_color)
        fill_direction = self._filament_fill_light_direction
        if isinstance(fill_color, (list, tuple)) and len(fill_color) >= 3:
            self._filament_fill_light_color = tuple(
                float(value) for value in fill_color[:3]
            )
        if isinstance(fill_direction, (list, tuple)) and len(fill_direction) >= 3:
            self._filament_fill_light_direction = tuple(
                float(value) for value in fill_direction[:3]
            )
        self._filament_fill_light_intensity = float(
            profile.get("controller_head_light_intensity", 1.0)
        )
        self._controller_hdr_lighting = bool(
            profile.get("controller_hdr_lighting", False)
        )
        if self._filament_lighting_presets:
            self._apply_filament_lighting_preset(
                self._filament_lighting_presets[
                    self._filament_lighting_preset_index
                ],
                apply_bridge=False,
            )
        screen = profile.get("screen")
        self._filament_screen_profile_authored = isinstance(screen, dict)
        if not isinstance(screen, dict):
            default_width = max(0.25, float(self.config.filament_screen_width))
            default_distance = max(0.25, float(self.config.filament_screen_distance))
            screen = {
                "position": [0.0, 0.0, -default_distance],
                "width": default_width,
                "height": default_width * 9.0 / 16.0,
                "rotation_deg": [0.0, 0.0, 0.0],
            }
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
        environment_lighting = (
            "hdr_ibl_pending_profile_fallback"
            if self._controller_hdr_lighting
            else "room_profile"
        )
        print(
            "Filament controller lighting: "
            f"environment={environment_lighting} "
            "screen_light=disabled",
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

    def _render_filament_multiview(
        self,
        render_views: list[Any],
        presentation_frame: VulkanStereoOutputFrame | None,
        acquired_image: tuple[_EyeSwapchain, int],
        finished_semaphore_available: bool,
        record_time: Callable[[str, float], None],
    ) -> int | None:
        bridge = self.filament_bridge
        eye, image_index = acquired_image
        state_started = time.perf_counter()
        bridge.set_active_eye(0)
        _update_filament_stereo_camera(
            bridge,
            render_views,
            near_plane=self._profile_near_plane,
            far_plane=self._profile_far_plane,
        )
        record_time("openxr_filament_multiview_state", state_started)
        bridge.set_active_eye(0)
        bridge.set_acquired_image(image_index)
        queue_started = time.perf_counter()
        bridge.begin_frame()
        record_time("openxr_filament_multiview_queue", queue_started)
        finish_started = time.perf_counter()
        bridge.end_frame()
        record_time("openxr_filament_multiview_finish_wait", finish_started)
        return (
            bridge.get_finished_drawing_semaphore()
            if finished_semaphore_available
            else None
        )

    def _render_projection_layer(
        self,
        views: list[Any],
        output_frame: VulkanStereoOutputFrame | None | object = _OUTPUT_FRAME_UNSET,
    ) -> Any | None:
        required_views = 2 if self._multiview_active else len(self.swapchains)
        if len(views) < required_views:
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
        with self._output_lock:
            sampling_frame = self._displayed_output
        if self._filament_animation_origin is None:
            self._filament_animation_origin = self._frame_now
        animation_time = max(0.0, self._frame_now - self._filament_animation_origin)
        acquired_images: list[tuple[_EyeSwapchain, int]] = []
        consumer_release_semaphores: list[int | None] = [None, None]
        consumer_completion_timeline: int | None = None
        submitted_filament_eyes: list[int] = []
        completion_drain_attempted = False
        finished_semaphore_available = False
        render_succeeded = False
        self._vulkan_projection_composer_active = False
        self._screen_quad_reprojection_active = False
        composer_frame = (
            output_frame
            if isinstance(output_frame, VulkanStereoOutputFrame)
            else sampling_frame
        )
        presentation_frame = (
            composer_frame
            if isinstance(composer_frame, VulkanStereoOutputFrame)
            else None
        )
        use_vulkan_projection_composer = bool(
            self._vulkan_projection_composer_requested
            and isinstance(composer_frame, VulkanStereoOutputFrame)
        )
        use_screen_quad_reprojection = bool(
            self._screen_quad_reprojection_requested
            and (
                self._can_use_screen_quad_reprojection(composer_frame)
                or bool(self._last_screen_quad_layers)
            )
        )
        if use_screen_quad_reprojection:
            use_vulkan_projection_composer = False
        filament_queue_lock = getattr(self.vulkan, "_lock", None)
        filament_queue_locked = False
        projection_started = time.perf_counter()

        def record_time(name: str, started: float) -> None:
            callback = self._on_breakdown_add_time
            if callback is not None:
                callback(name, max(0.0, time.perf_counter() - started))

        def prepare_filament_rendering() -> None:
            nonlocal filament_queue_locked, finished_semaphore_available
            if (
                self.filament_bridge is not None
                and filament_queue_lock is not None
                and not filament_queue_locked
            ):
                filament_lock_started = time.perf_counter()
                filament_queue_lock.acquire()
                filament_queue_locked = True
                record_time("openxr_filament_lock_wait", filament_lock_started)
            if self.filament_bridge is None:
                return
            # Glow is composed by VulkanProjectionScreenPass after the SBS
            # draw. Filament only owns environment/controller renderables.
            # Controller transforms and GLB animation state are shared by
            # both eye Views. Updating them twice adds owner-thread work
            # without changing either eye's scene state.
            self._update_filament_controllers(self.filament_bridge)
            if hasattr(self.filament_bridge, "apply_animations"):
                self.filament_bridge.apply_animations(animation_time)
            finished_semaphore_available = bool(
                getattr(
                    self.filament_bridge,
                    "finished_drawing_semaphore_abi_available",
                    False,
                )
            )

        try:
            # Acquire the complete stereo pair before entering either blocking
            # wait. This gives the runtime both requests up front and avoids
            # serializing the second acquire behind the first eye's wait.
            acquire_started = time.perf_counter()
            for eye in self.swapchains:
                image_index = xr.acquire_swapchain_image(eye.handle)
                acquired_images.append((eye, image_index))
            record_time("openxr_projection_acquire_pair", acquire_started)

            wait_pair_started = time.perf_counter()
            for eye_index, (eye, _image_index) in enumerate(acquired_images):
                wait_started = time.perf_counter()
                xr.wait_swapchain_image(
                    eye.handle,
                    xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                )
                record_time(
                    f"openxr_projection_wait_eye{eye_index}", wait_started
                )
            record_time("openxr_swapchain_wait", wait_pair_started)

            shared_prepare_started = time.perf_counter()
            if use_vulkan_projection_composer and presentation_frame is not None:
                sampling_frame = presentation_frame
            self._report_screen_resolution(views, presentation_frame)
            self._apply_screen_sampling_policy(
                presentation_frame
            )
            if (
                self.filament_bridge is not None
                and not use_vulkan_projection_composer
                and not use_screen_quad_reprojection
            ):
                prepare_filament_rendering()
            record_time("openxr_projection_shared_prepare", shared_prepare_started)

            if use_screen_quad_reprojection:
                try:
                    if isinstance(composer_frame, VulkanStereoOutputFrame):
                        self._render_screen_quad_reprojection(composer_frame)
                    elif not self._last_screen_quad_layers:
                        raise RuntimeError("screen Quad cache is empty")
                    else:
                        self._screen_quad_reprojection_active = True
                    self._clear_projection_targets(acquired_images)
                    render_succeeded = True
                except Exception as exc:
                    use_screen_quad_reprojection = False
                    self._screen_quad_reprojection_active = False
                    if self._on_breakdown_inc is not None:
                        self._on_breakdown_inc("openxr_screen_quad_fallback", 1)
                    self._report_screen_quad_reprojection_status(
                        "fallback", type(exc).__name__
                    )
                    use_vulkan_projection_composer = bool(
                        self._vulkan_projection_composer_requested
                        and isinstance(composer_frame, VulkanStereoOutputFrame)
                    )
            if not render_succeeded and use_vulkan_projection_composer:
                try:
                    composer_timeline = self._render_vulkan_projection_composer(
                        composer_frame,
                        acquired_images,
                        composition_views,
                    )
                    composer_frame.metadata["_vulkan_consumer_release_timeline"] = max(
                        int(
                            composer_frame.metadata.get(
                                "_vulkan_consumer_release_timeline", 0
                            )
                        ),
                        int(composer_timeline),
                    )
                    render_succeeded = True
                except Exception as exc:
                    use_vulkan_projection_composer = False
                    # The fallback renders only the Filament environment and
                    # controllers. The SBS image stays Composer-only.
                    self._vulkan_projection_composer_active = False
                    fallback_status = (type(exc).__name__, str(exc))
                    if fallback_status != self._last_vulkan_projection_composer_fallback:
                        self._last_vulkan_projection_composer_fallback = fallback_status
                        print(
                            "[OpenXRViewer] Vulkan projection composer fallback: "
                            f"{fallback_status[0]}: {fallback_status[1]}",
                            flush=True,
                        )
                    if self._on_breakdown_inc is not None:
                        self._on_breakdown_inc(
                            "openxr_vulkan_projection_composer_fallback", 1
                        )
                    fallback_prepare_started = time.perf_counter()
                    prepare_filament_rendering()
                    record_time(
                        "openxr_vulkan_composer_fallback_prepare",
                        fallback_prepare_started,
                    )
            if (
                not render_succeeded
                and self._multiview_active
                and self.filament_bridge is not None
            ):
                consumer_release_semaphores[0] = self._render_filament_multiview(
                    render_views,
                    presentation_frame,
                    acquired_images[0],
                    finished_semaphore_available,
                    record_time,
                )
                submitted_filament_eyes.append(0)
                render_succeeded = True
            if not render_succeeded:
                for eye_index, (eye, image_index) in enumerate(acquired_images):
                    if self.filament_bridge is not None:
                        bridge = self.filament_bridge
                        state_started = time.perf_counter()
                        bridge.set_active_eye(eye_index)
                        _update_filament_camera(
                            bridge,
                            render_views[eye_index],
                            near_plane=self._profile_near_plane,
                            far_plane=self._profile_far_plane,
                        )
                        record_time(
                            f"openxr_filament_eye{eye_index}_state", state_started
                        )
                        bridge.set_acquired_image(image_index)
                        queue_started = time.perf_counter()
                        bridge.begin_frame()
                        record_time(
                            f"openxr_filament_eye{eye_index}_queue", queue_started
                        )
                        finish_started = time.perf_counter()
                        bridge.end_frame()
                        record_time(
                            f"openxr_filament_eye{eye_index}_finish_wait",
                            finish_started,
                        )
                        submitted_filament_eyes.append(eye_index)
                        if finished_semaphore_available:
                            consumer_release_semaphores[eye_index] = (
                                bridge.get_finished_drawing_semaphore()
                            )
                    else:
                        image_address = _ctypes_handle_address(eye.images[image_index].image)
                        image = self.vulkan.image_handle_from_address(image_address)
                        self.vulkan.clear_color_image(image, self.config.clear_color)
            published_semaphores = tuple(
                semaphore
                for semaphore in consumer_release_semaphores
                if semaphore is not None
            )
            expected_semaphores = 1 if self._multiview_active else 2
            if use_vulkan_projection_composer:
                expected_semaphores = 0
            if finished_semaphore_available and len(published_semaphores) != expected_semaphores:
                raise RuntimeError(
                    "Filament did not publish the expected render-finished semaphores"
                )
            if published_semaphores:
                drain_started = time.perf_counter()
                completion_drain_attempted = True
                consumer_completion_timeline = self.vulkan.submit_on(
                    "graphics",
                    lambda _command_buffer: None,
                    wait_semaphore=published_semaphores,
                )
                record_time("openxr_filament_completion_drain", drain_started)
                if isinstance(sampling_frame, VulkanStereoOutputFrame):
                    sampling_frame.metadata["_vulkan_consumer_release_timeline"] = max(
                        int(
                            sampling_frame.metadata.get(
                                "_vulkan_consumer_release_timeline", 0
                            )
                        ),
                        int(consumer_completion_timeline),
                    )
            render_succeeded = True
            if presentation_frame is not None and (
                use_vulkan_projection_composer
                or use_screen_quad_reprojection
                or self.filament_bridge is None
            ):
                if self._on_breakdown_inc is not None:
                    self._on_breakdown_inc(
                        "openxr_new_screen_frame"
                        if isinstance(output_frame, VulkanStereoOutputFrame)
                        else "openxr_reused_screen_frame",
                        1,
                    )
            if self._on_breakdown_set_latest is not None:
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_composer_requested",
                    self._vulkan_projection_composer_requested,
                )
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_quality_chain_requested",
                    self._vulkan_projection_quality_chain_requested,
                )
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_composer_active",
                    self._vulkan_projection_composer_active,
                )
                self._on_breakdown_set_latest(
                    "openxr_vulkan_projection_composer_frame_id",
                    (
                        self._vulkan_projection_composer_frame_id
                        if self._vulkan_projection_composer_active
                        else -1
                    ),
                )
                self._on_breakdown_set_latest(
                    "openxr_screen_quad_reprojection_requested",
                    self._screen_quad_reprojection_requested,
                )
                self._on_breakdown_set_latest(
                    "openxr_screen_quad_reprojection_active",
                    self._screen_quad_reprojection_active,
                )
                self._on_breakdown_set_latest(
                    "openxr_projection_path",
                    (
                        "screen_quad_reprojection"
                        if self._screen_quad_reprojection_active
                        else (
                            "vulkan_composer"
                            if self._vulkan_projection_composer_active
                            else (
                                "filament_multiview"
                                if self._multiview_active
                                else "filament_per_eye"
                            )
                        )
                    ),
                )
        finally:
            if (
                not render_succeeded
                and self.filament_bridge is not None
                and submitted_filament_eyes
            ):
                try:
                    if not any(consumer_release_semaphores):
                        self.filament_bridge.wait_for_idle()
                    if finished_semaphore_available:
                        for eye_index in submitted_filament_eyes:
                            if consumer_release_semaphores[eye_index] is not None:
                                continue
                            self.filament_bridge.set_active_eye(eye_index)
                            consumer_release_semaphores[eye_index] = (
                                self.filament_bridge.get_finished_drawing_semaphore()
                            )
                    if (
                        not completion_drain_attempted
                        and any(consumer_release_semaphores)
                        and not bool(getattr(self.vulkan, "device_lost", False))
                    ):
                        completion_drain_attempted = True
                        consumer_completion_timeline = self.vulkan.submit_on(
                            "graphics",
                            lambda _command_buffer: None,
                            wait_semaphore=tuple(
                                semaphore
                                for semaphore in consumer_release_semaphores
                                if semaphore is not None
                            ),
                        )
                    if (
                        consumer_completion_timeline is not None
                        and isinstance(sampling_frame, VulkanStereoOutputFrame)
                    ):
                        sampling_frame.metadata[
                            "_vulkan_consumer_release_timeline"
                        ] = max(
                            int(
                                sampling_frame.metadata.get(
                                    "_vulkan_consumer_release_timeline", 0
                                )
                            ),
                            int(consumer_completion_timeline),
                        )
                except Exception:
                    pass
            if filament_queue_locked and filament_queue_lock is not None:
                filament_queue_lock.release()
                filament_queue_locked = False
            release_started = time.perf_counter()
            for eye, _image_index in acquired_images:
                xr.release_swapchain_image(eye.handle)
            record_time("openxr_projection_release_pair", release_started)
            if (
                isinstance(output_frame, VulkanStereoOutputFrame)
                and not render_succeeded
            ):
                self._abort_output_frame(output_frame)
            record_time("openxr_projection_total", projection_started)
        return OpenXrCompositionBuilder(xr, self.reference_space).projection_layer(
            composition_views, self.swapchains
        )

    @staticmethod
    def _save_visual_regression_host_image(
        host_image: VulkanHostImage,
        output_path: Path,
    ) -> None:
        """Save raw Vulkan pixels as an RGB diagnostic without changing them."""
        from PIL import Image

        pixels = host_image.read_rgba()
        vk = host_image.vk
        if int(host_image.format) in {
            int(vk.VK_FORMAT_B8G8R8A8_UNORM),
            int(vk.VK_FORMAT_B8G8R8A8_SRGB),
        }:
            pixels = pixels[..., [2, 1, 0, 3]]
        Image.fromarray(pixels[..., :3].copy(), mode="RGB").save(output_path)

    def _maybe_capture_visual_regression_frame(
        self,
        output_frame: VulkanStereoOutputFrame,
        *,
        eye_index: int,
        source_resource: VulkanImageResource | None,
        projection_resource: VulkanImageResource | None,
        projection_array_layer: int = 0,
        source_layout: int,
        source_access_mask: int,
        source_stage_mask: int,
    ) -> None:
        """Capture input/output/projection stages once from the live XR frame."""
        if self._visual_regression_capture_failed:
            return
        metadata = output_frame.metadata or {}
        output_dir_text = str(metadata.get("visual_regression_dir", "")).strip()
        if not output_dir_text:
            # Runtime visual regression is opt-in. Without producer metadata
            # there is no explicit capture request, so do not read back or
            # write projection images during normal rendering.
            return
        if not output_dir_text or self.vulkan is None:
            return
        eye = int(eye_index)
        if eye in self._visual_regression_capture_eyes:
            return
        if source_resource is None or projection_resource is None:
            self._visual_regression_capture_failed = True
            print(
                "[OpenXRViewer] visual regression capture skipped: "
                "production source or projection resource is unavailable",
                flush=True,
            )
            return
        try:
            from stereo_runtime.stage_visual_regression import _write_contact_sheet

            output_dir = Path(output_dir_text)
            output_dir.mkdir(parents=True, exist_ok=True)
            source_host_image = self._visual_regression_source_host_images.get(eye)
            if source_host_image is None:
                source_host_image = VulkanHostImage(
                    self.vulkan,
                    int(source_resource.width),
                    int(source_resource.height),
                    format=int(source_resource.format),
                    label=f"visual-regression-live-source-eye-{eye}",
                )
                self._visual_regression_source_host_images[eye] = source_host_image
            projection_host_image = self._visual_regression_projection_host_images.get(eye)
            if projection_host_image is None:
                projection_host_image = VulkanHostImage(
                    self.vulkan,
                    int(projection_resource.width),
                    int(projection_resource.height),
                    format=int(projection_resource.format),
                    label=f"visual-regression-live-projection-eye-{eye}",
                )
                self._visual_regression_projection_host_images[eye] = projection_host_image

            # Native Filament owns the projection render pass and the
            # producer may have registered the output state through a
            # different path. Normalize only the Python-side tracker for this
            # diagnostic copy; this does not submit a Vulkan barrier.
            vk = self.vulkan.vk
            self.vulkan.register_image_state(
                source_resource.image,
                ImageState(
                    layout=int(source_layout),
                    access_mask=int(source_access_mask),
                    stage_mask=int(source_stage_mask),
                    queue_family_index=self.vulkan.queue_family_index,
                ),
            )
            self.vulkan.register_image_state(
                projection_resource.image,
                ImageState(
                    layout=int(vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
                    access_mask=int(vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                    stage_mask=int(vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
                    queue_family_index=self.vulkan.queue_family_index,
                )
            )

            source_timeline = self.vulkan.copy_image(
                source_resource,
                source_host_image.resource,
            )
            self.vulkan.wait_for_timeline(source_timeline)
            self._save_visual_regression_host_image(
                source_host_image,
                output_dir / f"03_vulkan_output_{'left' if eye == 0 else 'right'}_eye.png",
            )

            try:
                projection_timeline = self.vulkan.copy_image(
                    projection_resource,
                    projection_host_image.resource,
                    source_array_layer=projection_array_layer,
                )
            except VulkanCapabilityError as first_exc:
                # Some OpenXR runtimes leave the Python-side swapchain state
                # stale after Filament's native render pass. Reassert the
                # known completed color-attachment state and retry once.
                self.vulkan.register_image_state(
                    projection_resource.image,
                    ImageState(
                        layout=int(vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
                        access_mask=int(vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT),
                        stage_mask=int(vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
                        queue_family_index=self.vulkan.queue_family_index,
                    ),
                )
                try:
                    projection_timeline = self.vulkan.copy_image(
                        projection_resource,
                        projection_host_image.resource,
                        source_array_layer=projection_array_layer,
                    )
                except Exception as retry_exc:
                    raise VulkanCapabilityError(
                        "projection image diagnostic copy failed after state retry: "
                        f"first={type(first_exc).__name__}: {first_exc}; "
                        f"retry={type(retry_exc).__name__}: {retry_exc}"
                    ) from retry_exc
            self.vulkan.wait_for_timeline(projection_timeline)
            self._save_visual_regression_host_image(
                projection_host_image,
                output_dir / f"06_openxr_projection_{'left' if eye == 0 else 'right'}_eye.png",
            )
            self._visual_regression_capture_eyes.add(eye)
            if len(self._visual_regression_capture_eyes) >= 2:
                manifest = {
                    "frame_id": int(output_frame.frame_id),
                    "source_stage": "vulkan_output_image",
                    "projection_stage": "openxr_projection_swapchain",
                    "readback": "temporary_host_image",
                    "color_space": str(output_frame.color_space),
                    "image_origin": str(output_frame.image_origin),
                    "vulkan_projection_quality_chain_requested": bool(
                        self._vulkan_projection_quality_chain_requested
                    ),
                    "source_size": [int(source_resource.width), int(source_resource.height)],
                    "projection_size": [int(projection_resource.width), int(projection_resource.height)],
                }
                (output_dir / "visual_regression_runtime_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _write_contact_sheet(output_dir)
                print(
                    f"[OpenXRViewer] automatic visual regression capture saved: {output_dir}",
                    flush=True,
                )
        except Exception as exc:
            self._visual_regression_capture_failed = True
            print(
                "[OpenXRViewer] automatic visual regression capture failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("OpenXrVulkanPresenter is not initialized")

    def _ensure_quad_swapchains(self, width: int, height: int) -> None:
        if self._quad_swapchain_extent == (width, height) and len(self._quad_swapchains) == 1:
            return
        if self.xr is None or self.session is None or self.vulkan is None:
            return
        self._destroy_quad_swapchains()
        vk = self.vulkan.vk
        formats = list(self.xr.enumerate_swapchain_formats(self.session))
        # The runtime output contract is display-referred sRGB. Match the
        # validated legacy Quad Layer path and prefer an sRGB target.
        quad_format = _select_swapchain_format(vk, formats, "srgb")
        handle = self.xr.create_swapchain(
            self.session,
            self.xr.SwapchainCreateInfo(
                usage_flags=(self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                             | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT),
                format=quad_format, sample_count=1, width=width, height=height,
                face_count=1, array_size=2, mip_count=1,
            ),
        )
        images = list(self.xr.enumerate_swapchain_images(
            handle, self.xr.SwapchainImageVulkan2KHR
        ))
        self._quad_swapchains.append(_EyeSwapchain(
            handle, images, width, height,
            self._register_swapchain_images(images, width, height, quad_format),
            array_size=2,
        ))
        self._quad_swapchain_format = int(quad_format)
        self._quad_swapchain_extent = (width, height)
        print(
            f"[OpenXRViewer] Quad layer swapchains created: "
            f"format={_vulkan_format_name(vk, quad_format)} extent={width}x{height} array_size=2",
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
                staging = entry.get("staging")
                if staging is not None:
                    staging.close()
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
        self._tool_quad_texture_cache.clear()
        self._tool_quad_texture_keys.clear()
        self._tool_overlay_xr_fps = 0.0
        self._tool_overlay_pending_xr_fps = 0.0
        self._tool_overlay_sbs_fps = 0.0
        self._tool_overlay_latency_ms = 0.0
        self._tool_overlay_depth_strength = 0.0
        self._tool_overlay_depth_strength_pending = None
        self._depth_osd_message = None
        self._tool_overlay_vr_res = (0, 0)
        self._tool_overlay_sbs_res = (0, 0)
        self._tool_overlay_pending_latency_ms = 0.0
        self._tool_overlay_xr_window_started = 0.0
        self._tool_overlay_xr_window_frames = 0
        self._tool_overlay_xr_frame_ts.clear()
        self._tool_overlay_sbs_window_started = 0.0
        self._tool_overlay_sbs_window_frames = 0
        self._tool_overlay_last_output_id = None

    def _update_tool_overlay_metrics(
        self, output_frame: VulkanStereoOutputFrame | None
    ) -> None:
        """Update low-rate overlay metrics without touching GPU resources."""
        now = float(self._frame_now or time.perf_counter())
        if output_frame is not None:
            depth_value = (getattr(output_frame, "metadata", None) or {}).get(
                "depth_strength"
            )
            try:
                depth_value = float(depth_value)
            except (TypeError, ValueError):
                depth_value = None
            if depth_value is not None and math.isfinite(depth_value):
                depth_value = max(0.0, depth_value)
                pending = self._tool_overlay_depth_strength_pending
                if pending is not None:
                    if abs(depth_value - pending) <= 1e-3:
                        self._tool_overlay_depth_strength_pending = None
                    else:
                        # Do not let an older in-flight output frame overwrite
                        # the value just accepted by the controller callback.
                        depth_value = None
                if depth_value is not None:
                    self._tool_overlay_depth_strength = depth_value
        if self._tool_overlay_xr_window_started <= 0.0:
            self._tool_overlay_xr_window_started = now
        self._tool_overlay_xr_window_frames += 1
        xr_elapsed = now - self._tool_overlay_xr_window_started
        if xr_elapsed >= _TOOL_OVERLAY_UPDATE_INTERVAL:
            if self._tool_overlay_pending_xr_fps > 0.0:
                self._tool_overlay_xr_fps = self._tool_overlay_pending_xr_fps
            else:
                self._tool_overlay_xr_fps = (
                    self._tool_overlay_xr_window_frames / xr_elapsed
                )
            # Keep all displayed performance values on the same low-rate
            # snapshot. Rebuilding the PIL texture from per-frame latency
            # defeats the legacy overlay cache and stalls the presenter.
            self._tool_overlay_latency_ms = self._tool_overlay_pending_latency_ms
            self._tool_overlay_xr_window_started = now
            self._tool_overlay_xr_window_frames = 0

        if self._tool_overlay_sbs_window_started <= 0.0:
            self._tool_overlay_sbs_window_started = now
        if output_frame is not None:
            frame_id = int(output_frame.frame_id)
            if frame_id != self._tool_overlay_last_output_id:
                self._tool_overlay_last_output_id = frame_id
                self._tool_overlay_sbs_window_frames += 1
                timestamp = float(output_frame.timestamp)
                latency_ms = (now - timestamp) * 1000.0
                if 0.0 <= latency_ms <= 10000.0:
                    self._tool_overlay_pending_latency_ms = latency_ms
        sbs_elapsed = now - self._tool_overlay_sbs_window_started
        if sbs_elapsed >= _TOOL_OVERLAY_UPDATE_INTERVAL:
            self._tool_overlay_sbs_fps = (
                self._tool_overlay_sbs_window_frames / sbs_elapsed
            )
            self._tool_overlay_sbs_window_started = now
            self._tool_overlay_sbs_window_frames = 0
        if callable(self._on_runtime_fps):
            try:
                runtime_fps = float(self._on_runtime_fps())
            except (TypeError, ValueError):
                runtime_fps = 0.0
            if math.isfinite(runtime_fps) and runtime_fps > 0.0:
                self._tool_overlay_sbs_fps = runtime_fps

    def _record_xr_presented_frame(self) -> None:
        timestamp = time.perf_counter()
        if self._on_breakdown_inc is not None:
            self._on_breakdown_inc("openxr_presented_frame", 1)
        self._tool_overlay_xr_frame_ts.append(timestamp)
        count = len(self._tool_overlay_xr_frame_ts)
        if count < 2:
            return
        span = timestamp - self._tool_overlay_xr_frame_ts[0]
        if span > 0.0:
            self._tool_overlay_pending_xr_fps = (count - 1) / span

    def _overlay_resolution_sizes(
        self, output_frame: VulkanStereoOutputFrame | None
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the live XR eye and per-eye output sizes for the FPS panel."""
        vr_res = tuple(self._tool_overlay_vr_res)
        if self.swapchains:
            eye = self.swapchains[0]
            candidate = (int(getattr(eye, "width", 0)), int(getattr(eye, "height", 0)))
            if candidate[0] > 0 and candidate[1] > 0:
                vr_res = candidate
                self._tool_overlay_vr_res = candidate

        sbs_res = tuple(self._tool_overlay_sbs_res)
        if output_frame is not None:
            metadata = dict(output_frame.metadata or {})
            candidate = metadata.get("render_size", metadata.get("source_render_size"))
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                candidate_size = (int(candidate[0]), int(candidate[1]))
                if candidate_size[0] > 0 and candidate_size[1] > 0:
                    sbs_res = candidate_size
                    self._tool_overlay_sbs_res = candidate_size
            if sbs_res == (0, 0):
                eye = getattr(output_frame, "left_eye", None)
                candidate_size = (
                    int(getattr(eye, "width", 0)),
                    int(getattr(eye, "height", 0)),
                )
                if candidate_size[0] > 0 and candidate_size[1] > 0:
                    sbs_res = candidate_size
                    self._tool_overlay_sbs_res = candidate_size
        return vr_res, sbs_res

    def _render_quad_layers(self, output_frame: VulkanStereoOutputFrame | None) -> list[Any]:
        # The main SBS screen is Projection Composer-only. Quad layers carry
        # controller tools and 2D overlays; they never replace the screen.
        return self._render_tool_quad_layers(output_frame)

    def _can_use_screen_quad_reprojection(
        self, frame: VulkanStereoOutputFrame | None
    ) -> bool:
        if (
            not isinstance(frame, VulkanStereoOutputFrame)
            or self._filament_screen is None
            or self.vulkan is None
            or self.xr is None
            or self.session is None
        ):
            return False
        metadata = frame.metadata or {}
        if not callable(metadata.get("_vulkan_source_prepare_for_sampling")):
            return False
        return all(
            int(getattr(source, "width", 0)) > 0
            and int(getattr(source, "height", 0)) > 0
            and getattr(source, "image", None) is not None
            for source in (frame.left_eye, frame.right_eye)
        )

    def _report_screen_quad_reprojection_status(self, *status: Any) -> None:
        current = tuple(status)
        if current == self._last_screen_quad_reprojection_status:
            return
        self._last_screen_quad_reprojection_status = current
        print(
            "[OpenXRViewer] Screen Quad Reprojection "
            + " ".join(str(value) for value in current),
            flush=True,
        )

    def _clear_projection_targets(
        self, acquired_images: list[tuple[_EyeSwapchain, int]]
    ) -> None:
        for eye_index, (eye, image_index) in enumerate(acquired_images):
            array_layers = range(eye.array_size) if len(acquired_images) == 1 else (0,)
            for array_layer in array_layers:
                self.vulkan.clear_color_image(
                    eye.resources[image_index].image,
                    self.config.clear_color,
                    base_array_layer=array_layer,
                )

    def _render_screen_quad_reprojection(
        self, frame: VulkanStereoOutputFrame
    ) -> None:
        if not self._can_use_screen_quad_reprojection(frame):
            raise RuntimeError("screen Quad prerequisites are unavailable")
        if self._screen_quad_reprojection_frame_id == int(frame.frame_id):
            self._screen_quad_reprojection_active = bool(self._last_screen_quad_layers)
            if self._screen_quad_reprojection_active and self._on_breakdown_inc is not None:
                self._on_breakdown_inc("openxr_screen_quad_reuse", 1)
            return
        width = int(frame.left_eye.width)
        height = int(frame.left_eye.height)
        if (int(frame.right_eye.width), int(frame.right_eye.height)) != (width, height):
            raise RuntimeError("stereo screen Quad source extents differ")
        upload_started = time.perf_counter()
        self._ensure_quad_swapchains(width, height)
        if len(self._quad_swapchains) != 1:
            raise RuntimeError("stereo screen Quad swapchains are unavailable")
        prepare_source = frame.metadata["_vulkan_source_prepare_for_sampling"]
        position, screen_width, screen_height, rotation = self._filament_screen
        diagnostic = _env_flag("D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC")
        screen_layers = []
        copy_timeline = 0
        quad_swapchain = self._quad_swapchains[0]
        with _acquired_swapchain_image(self.xr, quad_swapchain) as image_index:
            for eye_index, source in enumerate((frame.left_eye, frame.right_eye)):
                if diagnostic:
                    color = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0))[eye_index]
                    copy_timeline = max(
                        copy_timeline,
                        int(self.vulkan.clear_color_image(
                            quad_swapchain.resources[image_index].image,
                            color,
                            base_array_layer=eye_index,
                        )),
                    )
                else:
                    visible_semaphore = prepare_source(frame.frame_id, eye_index)
                    copy_timeline = max(
                        copy_timeline,
                        int(
                            self.vulkan.copy_image(
                                source,
                                quad_swapchain.resources[image_index],
                                wait_semaphore=visible_semaphore,
                                destination_array_layer=eye_index,
                                flip_y=False,
                            )
                        ),
                    )
                screen_layers.append(
                    OpenXrCompositionBuilder(self.xr, self.reference_space).quad_layer(
                        quad_swapchain, position, screen_width, screen_height, rotation, eye_index
                    )
                )
        frame.metadata["_vulkan_consumer_release_timeline"] = max(
            int(frame.metadata.get("_vulkan_consumer_release_timeline", 0)),
            copy_timeline,
        )
        # Quad swapchain images now own the submitted content. Return the
        # producer-owned source immediately instead of holding a runtime slot
        # until a later head-pose-only Quad reuse.
        self._release_output_frame(frame)
        with self._output_lock:
            previous = self._displayed_output
            if self._rendering_output is frame:
                self._rendering_output = None
            self._displayed_output = None
            if self._pending_output is frame:
                self._pending_output = None
        if previous is not None and previous is not frame:
            self._release_output_frame(previous)
        self._last_screen_quad_layers = screen_layers
        self._screen_quad_reprojection_frame_id = int(frame.frame_id)
        self._screen_quad_reprojection_active = True
        self._report_screen_quad_reprojection_status(
            "eye_diagnostic=left_red_right_green" if diagnostic else "active",
            f"source={width}x{height}",
        )
        if self._on_breakdown_inc is not None:
            self._on_breakdown_inc("openxr_screen_quad_new", 1)
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_screen_quad_upload", time.perf_counter() - upload_started
            )

    def _overlay_language(self) -> str:
        value = str(LANG or "EN").strip().upper()
        return "CN" if value.startswith(("CN", "ZH")) else "EN"

    def _filament_screen_pose_mat4(self) -> np.ndarray:
        position, _width, _height, rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        pose = euler_to_mat4(
            *(math.radians(float(value)) for value in rotation)
        ).astype(np.float64)
        pose[:3, 3] = np.asarray(position, dtype=np.float64)
        return pose

    @staticmethod
    def _overlay_pose_from_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
        quaternion = _mat3_to_quat_xyzw(matrix[:3, :3].astype(np.float64))
        position = tuple(float(value) for value in matrix[:3, 3])
        return position, tuple(float(value) for value in quaternion)

    def _screen_overlay_pose(self, local_model: np.ndarray):
        return self._overlay_pose_from_matrix(
            self._filament_screen_pose_mat4() @ local_model
        )

    def _controller_overlay_pose(self, hand: int, panel_height: float, top_ref: float):
        grip = self._grip_mat_l if int(hand) == 0 else self._grip_mat_r
        aim = self._aim_mat_l if int(hand) == 0 else self._aim_mat_r
        panel_pos = panel_fwd = panel_up = None
        if grip is not None and aim is not None:
            grip_up = np.asarray(grip[:3, 1], dtype=np.float64)
            grip_up /= max(float(np.linalg.norm(grip_up)), 1e-10)
            laser_fwd = -np.asarray(aim[:3, 2], dtype=np.float64)
            right_axis = np.asarray(aim[:3, 0], dtype=np.float64)
            right_axis /= max(float(np.linalg.norm(right_axis)), 1e-10)
            angle = math.radians(12.0)
            laser_fwd = (
                laser_fwd * math.cos(angle)
                + np.cross(right_axis, laser_fwd) * math.sin(angle)
                + right_axis * float(np.dot(right_axis, laser_fwd))
                * (1.0 - math.cos(angle))
            )
            laser_fwd /= max(float(np.linalg.norm(laser_fwd)), 1e-10)
            grip_pos = np.asarray(grip[:3, 3], dtype=np.float64)
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11
            panel_fwd = grip_up - laser_fwd
            panel_fwd /= max(float(np.linalg.norm(panel_fwd)), 1e-10)
            panel_up = grip_up
            panel_right = np.cross(panel_up, panel_fwd)
            panel_right /= max(float(np.linalg.norm(panel_right)), 1e-10)
            panel_up2 = np.cross(panel_fwd, panel_right)
            panel_up2 /= max(float(np.linalg.norm(panel_up2)), 1e-10)
            panel_pos = (
                laser_origin
                + panel_fwd * 0.05
                + panel_up2 * (top_ref - panel_height / 2.0)
            )
            basis = np.column_stack((panel_right, panel_up2, panel_fwd))
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = basis
            matrix[:3, 3] = panel_pos
            return self._overlay_pose_from_matrix(matrix)

        if self._head_position_w is None or self._head_forward_w is None:
            return None
        head = np.asarray(self._head_position_w, dtype=np.float64)
        forward = np.asarray(self._head_forward_w, dtype=np.float64)
        forward /= max(float(np.linalg.norm(forward)), 1e-10)
        panel_pos = head + forward * (1.0 if int(hand) == 0 else 1.2)
        panel_pos[1] += -0.15 if int(hand) == 0 else -0.3
        panel_fwd = -forward
        panel_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        panel_right = np.cross(panel_up, panel_fwd)
        panel_right /= max(float(np.linalg.norm(panel_right)), 1e-10)
        panel_up2 = np.cross(panel_fwd, panel_right)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.column_stack((panel_right, panel_up2, panel_fwd))
        matrix[:3, 3] = panel_pos
        return self._overlay_pose_from_matrix(matrix)

    def _operation_guide_environment_mode(self) -> bool:
        name = str(getattr(self, "_profile_view_name", "") or "").strip().lower()
        return bool(name and name not in {"default", "none"})

    def _cursor_overlay_specs(self, rgba, screen_pose, head):
        """Build the legacy laser hit rings as transparent tool quads."""
        specs = []
        for hand in (0, 1):
            origin, direction = self._controller_interaction_ray(hand)
            if origin is None or direction is None:
                continue
            keyboard_hit = None
            if self._keyboard_visible:
                keyboard_hit = self._keyboard_plane_hit(origin, direction)
            if keyboard_hit != (None, None) and keyboard_hit is not None:
                pose = self._keyboard_pose_mat4()
                x, y = (float(keyboard_hit[0]), float(keyboard_hit[1]))
                local = np.asarray((x, y, 0.0), dtype=np.float64)
                matrix = pose.copy()
                matrix[:3, 3] = (
                    pose[:3, 3] + pose[:3, :3] @ local + pose[:3, 2] * 0.003
                )
            else:
                hit = self._screen_ray_hit_for_hand(hand)
                if hit is None:
                    continue
                u, v = (float(hit[0]), float(hit[1]))
                _position, width, height, _rotation = self._filament_screen or (
                    (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
                )
                matrix = screen_pose.copy()
                if self._screen_curved:
                    half_angle = min(0.72, math.pi / 2.0)
                    angle = -half_angle + 2.0 * half_angle * u
                    tangent = np.asarray(
                        (math.cos(angle), 0.0, math.sin(angle)),
                        dtype=np.float64,
                    )
                    normal = np.asarray(
                        (-math.sin(angle), 0.0, math.cos(angle)),
                        dtype=np.float64,
                    )
                    curved_basis = np.column_stack(
                        (tangent, np.asarray((0.0, 1.0, 0.0)), normal)
                    )
                    matrix[:3, :3] = screen_pose[:3, :3] @ curved_basis
                    hit_world = self._screen_uv_to_world(u, v)
                    if hit_world is None:
                        continue
                    matrix[:3, 3] = (
                        hit_world + screen_pose[:3, :3] @ normal * 0.003
                    )
                else:
                    matrix[:3, 3] = (
                        screen_pose[:3, 3]
                        + screen_pose[:3, :3]
                        @ np.asarray(((u - 0.5) * float(width),
                                      (v - 0.5) * float(height), 0.0), dtype=np.float64)
                        + screen_pose[:3, 2] * 0.003
                    )
            distance = float(np.linalg.norm(matrix[:3, 3] - head))
            radius = 0.012 * float(np.clip(distance / 2.0, 0.35, 50.0))
            position, rotation = self._overlay_pose_from_matrix(matrix)
            specs.append(
                (
                    f"laser_cursor_{hand}",
                    rgba,
                    position,
                    (radius * 2.0, radius * 2.0),
                    rotation,
                )
            )
        return specs

    def _render_tool_quad_layers(
        self, output_frame: VulkanStereoOutputFrame | None = None
    ) -> list[Any]:
        """Submit the legacy keyboard and overlay quads with legacy poses."""
        if self.xr is None or self.session is None or self.vulkan is None:
            return []
        _position, width, height, _rotation = self._filament_screen or (
            (0.0, 1.2, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
        )
        width = float(width)
        height = float(height)
        vr_res, sbs_res = self._overlay_resolution_sizes(output_frame)
        language = self._overlay_language()
        environment_mode = self._operation_guide_environment_mode()
        specs = []

        if self._keyboard_visible:
            keyboard_width = float(self._keyboard_width)
            keyboard_height = float(self._keyboard_height)
            hover_indices = tuple(
                sorted(
                    index
                    for index in (self._kb_hover_l, self._kb_hover_r)
                    if index is not None
                )
            )
            held_indices = tuple(
                sorted(
                    index
                    for index in (self._kb_held_key_l, self._kb_held_key_r)
                    if index is not None
                )
            )
            modifier_vks = {0x10: "shift", 0x11: "ctrl", 0x12: "alt", 0x5B: "win"}
            locked_indices = [
                index
                for index, key in enumerate(self._keyboard_keys)
                if key.vk in modifier_vks
                and bool(self._mod_state[modifier_vks[key.vk]][0])
            ]
            if self._caps_lock:
                locked_indices.extend(
                    index
                    for index, key in enumerate(self._keyboard_keys)
                    if key.vk == 0x14
                )
            locked_indices = tuple(sorted(set(locked_indices)))
            keyboard_cache_key = (
                bool(self._kb_show_shifted), keyboard_width, keyboard_height,
                hover_indices, held_indices, locked_indices,
            )
            rgba = self._tool_quad_texture_cache.get("keyboard")
            if rgba is None or self._tool_quad_texture_keys.get("keyboard") != keyboard_cache_key:
                rgba, self._keyboard_keys = build_keyboard_rgba(
                    self._kb_show_shifted,
                    keyboard_width,
                    keyboard_height,
                    hover_indices=hover_indices,
                    held_indices=held_indices,
                    locked_indices=locked_indices,
                )
                self._tool_quad_texture_cache["keyboard"] = rgba
                self._tool_quad_texture_keys["keyboard"] = keyboard_cache_key
            keyboard_pose = self._keyboard_pose_mat4()
            _keyboard_position, keyboard_quaternion = self._overlay_pose_from_matrix(
                keyboard_pose
            )
            specs.append(
                (
                    "keyboard", rgba, _keyboard_position,
                    (keyboard_width, keyboard_height), keyboard_quaternion,
                )
            )

        head = np.asarray(
            self._head_position_w if self._head_position_w is not None else (0, 0, 0),
            dtype=np.float64,
        )
        screen_pose = self._filament_screen_pose_mat4()
        screen_distance = float(np.linalg.norm(screen_pose[:3, 3] - head))
        now = float(self._frame_now or time.perf_counter())
        preset_osd_active = (
            self._preset_name_overlay
            and now - float(self._preset_osd_show_t) < 5.0
        )
        screen_osd_active = now - float(self._screen_osd_show_t) < 2.5
        if preset_osd_active or screen_osd_active:
            # Match the legacy screen OSD: dark rounded panel, grey labels,
            # cyan values, and a centered text group.
            if screen_osd_active:
                msdf_atlas = self._msdf_font_atlas
                osd_key = (
                    "screen_adjust_osd",
                    "gpu-msdf"
                    if self._vulkan_msdf_quad_renderer is not None
                    else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
                    round(width, 2),
                    round(screen_distance, 2),
                )
                osd_rgba = self._tool_quad_texture_cache.get("screen_osd")
                if (
                    osd_rgba is None
                    or self._tool_quad_texture_keys.get("screen_osd") != osd_key
                ):
                    if msdf_atlas is not None:
                        runs = (
                            ("Size", (150, 158, 185, 255)),
                            (
                                f"{width:.2f} x {width * 9.0 / 16.0:.2f} m",
                                (0, 210, 230, 255),
                            ),
                            ("Dist", (150, 158, 185, 255)),
                            (f"{screen_distance:.2f} m", (0, 210, 230, 255)),
                        )
                        canvas_width, canvas_height, msdf_runs = _layout_msdf_osd_runs(
                            msdf_atlas, runs
                        )
                        if self._vulkan_msdf_quad_renderer is not None:
                            osd_rgba = VulkanMsdfQuadRequest(
                                width=canvas_width,
                                height=canvas_height,
                                runs=msdf_runs,
                            )
                        else:
                            from .overlay_textures import build_msdf_text_osd_rgba

                            osd_rgba = build_msdf_text_osd_rgba(
                                msdf_atlas,
                                size=(canvas_width, canvas_height),
                                runs=msdf_runs,
                            )
                    else:
                        canvas_width, canvas_height = 512, 78
                        osd_rgba = build_screen_adjust_osd_rgba(
                            width,
                            screen_distance,
                            size=(canvas_width, canvas_height),
                        )
                    self._tool_quad_texture_cache["screen_osd"] = osd_rgba
                    self._tool_quad_texture_keys["screen_osd"] = osd_key
                if isinstance(osd_rgba, VulkanMsdfQuadRequest):
                    canvas_width, canvas_height = osd_rgba.width, osd_rgba.height
                elif hasattr(osd_rgba, "shape"):
                    canvas_height, canvas_width = osd_rgba.shape[:2]
                else:
                    canvas_width, canvas_height = 512, 78
                osd_height = width * 0.03 * (
                    float(canvas_height) / _MSDF_OSD_REFERENCE_HEIGHT
                )
                osd_width = osd_height * (
                    float(canvas_width) / max(1.0, float(canvas_height))
                )
            else:
                msdf_atlas = self._msdf_font_atlas
                osd_key = (
                    "preset_osd",
                    "gpu-msdf"
                    if self._vulkan_msdf_quad_renderer is not None
                    else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
                    str(self._preset_name_overlay),
                    round(width, 3),
                    round(height, 3),
                )
                osd_rgba = self._tool_quad_texture_cache.get("screen_osd")
                if (
                    osd_rgba is None
                    or self._tool_quad_texture_keys.get("screen_osd") != osd_key
                ):
                    if msdf_atlas is not None:
                        label = "Preset"
                        value = str(self._preset_name_overlay)
                        runs = (
                            {
                                "text": label,
                                "color": (150, 158, 185, 255),
                            },
                            {
                                "text": value,
                                "color": (0, 210, 230, 255),
                            },
                        )
                        canvas_width, canvas_height, msdf_runs = _layout_msdf_osd_runs(
                            msdf_atlas, runs
                        )
                        if self._vulkan_msdf_quad_renderer is not None:
                            osd_rgba = VulkanMsdfQuadRequest(
                                width=canvas_width,
                                height=canvas_height,
                                runs=msdf_runs,
                            )
                        else:
                            from .overlay_textures import build_msdf_text_osd_rgba

                            osd_rgba = build_msdf_text_osd_rgba(
                                msdf_atlas,
                                size=(canvas_width, canvas_height),
                                runs=msdf_runs,
                            )
                    else:
                        canvas_width, canvas_height = 768, 78
                        osd_rgba = build_screen_preset_osd_rgba(
                            str(self._preset_name_overlay),
                            size=(canvas_width, canvas_height),
                        )
                    self._tool_quad_texture_cache["screen_osd"] = osd_rgba
                    self._tool_quad_texture_keys["screen_osd"] = osd_key
                if isinstance(osd_rgba, VulkanMsdfQuadRequest):
                    canvas_width, canvas_height = osd_rgba.width, osd_rgba.height
                elif hasattr(osd_rgba, "shape"):
                    canvas_height, canvas_width = osd_rgba.shape[:2]
                else:
                    canvas_width, canvas_height = 768, 78
                osd_height = width * 0.03 * (
                    float(canvas_height) / _MSDF_OSD_REFERENCE_HEIGHT
                )
                osd_width = osd_height * (
                    float(canvas_width) / max(1.0, float(canvas_height))
                )
            osd_local = np.eye(4, dtype=np.float64)
            # Keep the legacy gap between the screen edge and the centered OSD.
            osd_local[:3, 3] = (
                0.0,
                height / 2.0 + width * 0.016 + osd_height / 2.0,
                0.0,
            )
            osd_position, osd_rotation = self._screen_overlay_pose(osd_local)
            specs.append(
                (
                    "screen_osd",
                    osd_rgba,
                    osd_position,
                    (osd_width, osd_height),
                    osd_rotation,
                )
            )
        depth_osd_active = now - float(self._depth_osd_show_t) < 2.5
        if depth_osd_active:
            msdf_atlas = self._msdf_font_atlas
            depth_key = (
                "depth_osd",
                "gpu-msdf"
                if self._vulkan_msdf_quad_renderer is not None
                else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
                round(self._tool_overlay_depth_strength, 3),
                self._depth_osd_message,
            )
            depth_rgba = self._tool_quad_texture_cache.get("depth_osd")
            if (
                depth_rgba is None
                or self._tool_quad_texture_keys.get("depth_osd") != depth_key
            ):
                if msdf_atlas is not None:
                    depth_request = _build_msdf_depth_osd_request(
                        msdf_atlas,
                        self._tool_overlay_depth_strength,
                        self._depth_osd_message,
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        depth_rgba = depth_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        depth_rgba = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(depth_request.width, depth_request.height),
                            runs=depth_request.runs,
                            background=depth_request.background,
                            radius=int(depth_request.radius),
                        )
                else:
                    from .overlay_textures import build_short_osd_rgba

                    depth_rgba = build_short_osd_rgba(
                        [
                            self._depth_osd_message
                            or (
                                f"Depth Strength "
                                f"{self._tool_overlay_depth_strength:.2f}"
                            )
                        ]
                    )
                self._tool_quad_texture_cache["depth_osd"] = depth_rgba
                self._tool_quad_texture_keys["depth_osd"] = depth_key
            if isinstance(depth_rgba, VulkanMsdfQuadRequest):
                depth_canvas_width, depth_canvas_height = (
                    depth_rgba.width,
                    depth_rgba.height,
                )
            else:
                depth_canvas_height, depth_canvas_width = depth_rgba.shape[:2]
            depth_osd_height = width * 0.03 * (
                float(depth_canvas_height) / _MSDF_OSD_REFERENCE_HEIGHT
            )
            depth_osd_width = depth_osd_height * (
                float(depth_canvas_width) / max(1.0, float(depth_canvas_height))
            )
            depth_local = np.eye(4, dtype=np.float64)
            depth_local[:3, 3] = (
                0.0,
                height / 2.0 + width * 0.016 + depth_osd_height / 2.0,
                0.0,
            )
            depth_position, depth_rotation = self._screen_overlay_pose(depth_local)
            specs.append(
                (
                    "depth_osd",
                    depth_rgba,
                    depth_position,
                    (depth_osd_width, depth_osd_height),
                    depth_rotation,
                )
            )
        msdf_atlas = self._msdf_font_atlas
        fps_key = (
            "fps", language, environment_mode, round(width, 3), round(height, 3),
            round(self._tool_overlay_xr_fps, 1), round(self._tool_overlay_sbs_fps, 1),
            round(self._tool_overlay_latency_ms, 1),
            round(self._tool_overlay_depth_strength, 3),
            vr_res,
            sbs_res,
        )
        if self._fps_overlay_visible or self._hand_fps_visible:
            rgba = self._tool_quad_texture_cache.get("fps")
            if rgba is None or self._tool_quad_texture_keys.get("fps") != fps_key:
                if msdf_atlas is not None:
                    msdf_request = _build_msdf_fps_panel(
                        msdf_atlas,
                        actual_fps=self._tool_overlay_xr_fps,
                        sbs_fps=self._tool_overlay_sbs_fps,
                        latency_ms=self._tool_overlay_latency_ms,
                        screen_width=width,
                        screen_height=height,
                        screen_distance=screen_distance,
                        depth_strength=self._tool_overlay_depth_strength,
                        vr_res=vr_res,
                        sbs_res=sbs_res,
                        controller_brand=getattr(self._controller_brand, "name", ""),
                        environment_visible=environment_mode,
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        rgba = msdf_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        rgba = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(msdf_request.width, msdf_request.height),
                            runs=msdf_request.runs,
                            background=msdf_request.background,
                            radius=int(msdf_request.radius),
                        )
                else:
                    rgba = build_fps_overlay_rgba(
                        actual_fps=self._tool_overlay_xr_fps,
                        sbs_fps=self._tool_overlay_sbs_fps,
                        latency_ms=self._tool_overlay_latency_ms,
                        screen_width=width,
                        screen_height=height,
                        screen_distance=screen_distance,
                        depth_strength=self._tool_overlay_depth_strength,
                        vr_res=vr_res,
                        sbs_res=sbs_res,
                        controller_brand=getattr(self._controller_brand, "name", ""),
                        environment_visible=environment_mode,
                    )
                self._tool_quad_texture_cache["fps"] = rgba
                self._tool_quad_texture_keys["fps"] = fps_key

        if self._fps_overlay_visible:
            overlay_h = height / 8.0
            overlay_w = overlay_h * float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))
            local = np.eye(4, dtype=np.float64)
            local[:3, 3] = (
                -width / 2.0 + overlay_w / 2.0,
                -height / 2.0 - height * 0.02 - overlay_h / 2.0,
                0.0,
            )
            fps_position, fps_rotation = self._screen_overlay_pose(local)
            specs.append(("screen_fps", rgba, fps_position, (overlay_w, overlay_h), fps_rotation))

        if self._screen_operation_guide_visible:
            # Keep the guide's glyph layout proportional to its canvas. The
            # whole Quad already follows the screen height, so shrinking the
            # glyphs inversely with a large screen would leave a tiny island of
            # text in the middle of an otherwise empty guide panel.
            screen_guide_scale = 1.0
            help_key = (
                "screen_help",
                language,
                round(screen_guide_scale, 4),
                "gpu-msdf"
                if self._vulkan_msdf_quad_renderer is not None
                else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
            )
            rgba = self._tool_quad_texture_cache.get("screen_help")
            if rgba is None or self._tool_quad_texture_keys.get("screen_help") != help_key:
                # This is the legacy screen-side vertical guide. The
                # controller-attached two-column guide is a different panel.
                if msdf_atlas is not None:
                    rows, _env_rows = get_controller_help_rows(language)
                    msdf_request = _build_msdf_help_panel(
                        msdf_atlas,
                        rows,
                        two_columns=False,
                        size_scale=screen_guide_scale,
                        canvas_scale=1.0,
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        rgba = msdf_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        rgba = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(msdf_request.width, msdf_request.height),
                            runs=msdf_request.runs,
                            background=msdf_request.background,
                            radius=int(msdf_request.radius),
                        )
                else:
                    rgba = build_team_help_rgba(lang=language)
                self._tool_quad_texture_cache["screen_help"] = rgba
                self._tool_quad_texture_keys["screen_help"] = help_key
            # The panel always follows the screen height. Its MSDF texture
            # layout is scaled independently so text remains proportional.
            panel_h = height
            panel_w = panel_h * float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))
            gap = height * 0.02
            head_local = screen_pose[:3, :3].T @ (head - screen_pose[:3, 3])
            hinge = np.asarray((-width / 2.0 - gap, 0.0, 0.0), dtype=np.float64)
            to_user = head_local - hinge
            to_user /= max(float(np.linalg.norm(to_user)), 1e-10)
            theta = math.atan2(float(to_user[0]), float(to_user[2]))
            hinge_rotation = np.eye(4, dtype=np.float64)
            hinge_rotation[0, 0] = math.cos(theta)
            hinge_rotation[0, 2] = math.sin(theta)
            hinge_rotation[2, 0] = -math.sin(theta)
            hinge_rotation[2, 2] = math.cos(theta)
            hinge_translation = np.eye(4, dtype=np.float64)
            hinge_translation[0, 3] = -width / 2.0 - gap
            panel_offset = np.eye(4, dtype=np.float64)
            panel_offset[0, 3] = -panel_w / 2.0
            panel_position, panel_rotation = self._overlay_pose_from_matrix(
                screen_pose @ hinge_translation @ hinge_rotation @ panel_offset
            )
            specs.append(("screen_help", rgba, panel_position, (panel_w, panel_h), panel_rotation))

        if self._hand_fps_visible:
            pose = self._controller_overlay_pose(0, 0.075, 0.10)
            if pose is not None:
                hand_position, hand_rotation = pose
                overlay_w = 0.075 * float(rgba.shape[1]) / max(1.0, float(rgba.shape[0]))
                specs.append(("hand_fps", rgba, hand_position, (overlay_w, 0.075), hand_rotation))

        if self._hand_operation_guide_visible:
            help_key = (
                "hand_help",
                language,
                environment_mode,
                "gpu-msdf"
                if self._vulkan_msdf_quad_renderer is not None
                else ("cpu-msdf" if msdf_atlas is not None else "legacy"),
            )
            hand_help = self._tool_quad_texture_cache.get("hand_help")
            if hand_help is None or self._tool_quad_texture_keys.get("hand_help") != help_key:
                if msdf_atlas is not None:
                    rows, env_rows = get_controller_help_rows(language)
                    selected_rows = env_rows if environment_mode else rows
                    msdf_request = _build_msdf_help_panel(
                        msdf_atlas, selected_rows, two_columns=True
                    )
                    if self._vulkan_msdf_quad_renderer is not None:
                        hand_help = msdf_request
                    else:
                        from .overlay_textures import build_msdf_text_osd_rgba

                        hand_help = build_msdf_text_osd_rgba(
                            msdf_atlas,
                            size=(msdf_request.width, msdf_request.height),
                            runs=msdf_request.runs,
                            background=msdf_request.background,
                            radius=int(msdf_request.radius),
                        )
                else:
                    hand_help = build_help_rgba(environment_mode=environment_mode, lang=language)
                self._tool_quad_texture_cache["hand_help"] = hand_help
                self._tool_quad_texture_keys["hand_help"] = help_key
            panel_h = 0.2
            panel_w = panel_h * float(hand_help.shape[1]) / max(1.0, float(hand_help.shape[0]))
            pose = self._controller_overlay_pose(1, panel_h, panel_h + 0.025)
            if pose is not None:
                hand_position, hand_rotation = pose
                specs.append(("hand_help", hand_help, hand_position, (panel_w, panel_h), hand_rotation))

        if self._aperture_visible:
            aperture_key = ("Aperture", "B: close", 384, 64)
            rgba = self._tool_quad_texture_cache.get("aperture")
            if rgba is None or self._tool_quad_texture_keys.get("aperture") != aperture_key:
                rgba = build_short_osd_rgba(("Aperture", "B: close"), width=384, height=64)
                self._tool_quad_texture_cache["aperture"] = rgba
                self._tool_quad_texture_keys["aperture"] = aperture_key
            position, rotation = self._overlay_pose_from_matrix(screen_pose)
            specs.append(("aperture", rgba, position, (width * 0.24, height * 0.06), rotation))

        cursor_rgba = self._tool_quad_texture_cache.get("laser_cursor")
        if cursor_rgba is None:
            cursor_rgba = build_cursor_rgba(64)
            self._tool_quad_texture_cache["laser_cursor"] = cursor_rgba
            self._tool_quad_texture_keys["laser_cursor"] = ("legacy_cursor_ring", 64)
        specs.extend(self._cursor_overlay_specs(cursor_rgba, screen_pose, head))

        specs_ready = time.perf_counter()
        layers = [self._upload_tool_quad(*spec) for spec in specs]
        if self._on_breakdown_add_time is not None:
            self._on_breakdown_add_time(
                "openxr_quad_upload", time.perf_counter() - specs_ready
            )
        return layers

    def _cache_head_position(self, views: list[Any]) -> None:
        if len(views) < 2:
            self._head_position_w = None
            self._head_forward_w = None
            return
        eye_positions = [
            np.asarray(
                (view.pose.position.x, view.pose.position.y, view.pose.position.z),
                dtype=np.float64,
            )
            for view in views[:2]
        ]
        self._head_position_w = (eye_positions[0] + eye_positions[1]) * 0.5
        head_matrix = _xr_view_pose_to_model_mat4(views[0].pose)
        self._head_forward_w = -head_matrix[:3, 2].astype(np.float64)
        if self._head_position_w is not None:
            self._initial_head_y = float(self._head_position_w[1])

    def _report_profile_alignment(self) -> None:
        """Log the calibrated head pose against the authored GLB-local target once."""
        if self._profile_alignment_logged:
            return
        if self._profile_head_transform is None or self._head_position_w is None:
            return
        target = np.asarray(self._profile_head_transform[:3, 3], dtype=np.float64)
        actual = np.asarray(self._head_position_w, dtype=np.float64)
        delta = actual - target
        self._profile_alignment_logged = True
        print(
            "[OpenXRViewer] profile head alignment: "
            f"target_glb=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f}) "
            f"actual_xr=({actual[0]:.3f},{actual[1]:.3f},{actual[2]:.3f}) "
            f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) "
            f"reference_space={getattr(self._reference_space_type, 'name', self._reference_space_type)}",
            flush=True,
        )

    def _initialize_filament_screen_from_head(self) -> None:
        """Initialize an unauthored screen with the legacy head-centered preset."""
        if (
            self._filament_screen_head_initialized
            or self._filament_screen_profile_authored
            or self._filament_screen is None
            or self._head_position_w is None
            or self._head_forward_w is None
        ):
            return
        self._shortcut_screen_preset_index = 5
        self._apply_shortcut_screen_preset(5)
        self._filament_screen_initial = self._filament_screen
        self._filament_screen_head_initialized = True

    def _controller_guide_geometry(self):
        """Return the world-space panel geometry for the Projection Layer guide."""
        if self._grip_mat_r is None or self._head_position_w is None:
            return None
        controller_position = np.asarray(self._grip_mat_r[:3, 3], dtype=np.float64)
        to_head = np.asarray(self._head_position_w, dtype=np.float64) - controller_position
        distance = float(np.linalg.norm(to_head))
        if distance <= 1e-6 or distance > self.config.controller_guide_max_distance:
            return None

        def normalized(vector):
            vector = np.asarray(vector, dtype=np.float64)
            return vector / max(float(np.linalg.norm(vector)), 1e-6)

        button_position = self._controller_b_button_world_position()
        if button_position is None:
            button_position = controller_position
        forward = normalized(np.asarray(self._head_position_w, dtype=np.float64) - button_position)
        world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        right = normalized(np.cross(world_up, forward))
        up = normalized(np.cross(forward, right))

        # Keep the Quad head-facing while solving its center from the B button
        # world position and the callout endpoint's local texture coordinate.
        endpoint_x = (540.0 / 1024.0 - 0.5) * 0.34
        endpoint_y = (0.5 - 300.0 / 768.0) * 0.255
        panel_position = (
            button_position
            - right * endpoint_x
            - up * endpoint_y
            + forward * 0.006
        )
        basis = np.column_stack((right, up, forward))
        return (
            tuple(float(value) for value in panel_position),
            (0.34, 0.255),
            basis,
        )

    def _controller_guide_pose(self):
        """Return the legacy pose representation used by geometry tests."""
        geometry = self._controller_guide_geometry()
        if geometry is None:
            return None
        position, size, basis = geometry
        rotation = _mat3_to_quat_xyzw(basis)
        return (
            position,
            size,
            tuple(float(value) for value in rotation),
        )

    def _resolve_controller_b_button_local(
        self, *, force: bool = False
    ) -> np.ndarray | None:
        if force:
            controller_button_local_position.cache_clear()
            self._controller_b_button_local = None
            self._controller_b_button_resolved = False
        if self._controller_brand is None:
            self._controller_b_button_local = None
            self._controller_b_button_resolved = True
            return None
        if not self._controller_b_button_resolved:
            resolved = controller_button_local_position(
                str(self._controller_brand.right_glb), "b_button"
            )
            self._controller_b_button_local = (
                None if resolved is None else np.asarray(resolved, dtype=np.float64)
            )
            self._controller_b_button_resolved = True
        return self._controller_b_button_local

    def _controller_b_button_world_position(self):
        if self._grip_mat_r is None:
            return None
        button_local = self._resolve_controller_b_button_local()
        if button_local is None:
            return None

        offset = np.eye(4, dtype=np.float64)
        offset[:3, 3] = np.asarray(
            self._controller_calibration_offset, dtype=np.float64
        )
        rotation = euler_to_mat4(
            0.0, math.radians(self._controller_calibration_rotation_deg), 0.0
        ).astype(np.float64)
        model_matrix = np.asarray(self._grip_mat_r, dtype=np.float64) @ rotation @ offset
        local = np.ones(4, dtype=np.float64)
        local[:3] = button_local
        return (model_matrix @ local)[:3]

    def _upload_msdf_tool_quad(self, key, request, position, size, rotation):
        renderer = self._vulkan_msdf_quad_renderer
        if renderer is None:
            raise RuntimeError("Vulkan MSDF Quad renderer is unavailable")
        height, width = int(request.height), int(request.width)
        format_value = self._tool_quad_format()
        if not renderer.supports_destination_format(format_value):
            from .overlay_textures import build_msdf_text_osd_rgba

            return self._upload_tool_quad(
                key,
                build_msdf_text_osd_rgba(
                    self._msdf_font_atlas,
                    size=(width, height),
                    runs=request.runs,
                    background=request.background,
                    radius=int(request.radius),
                ),
                position,
                size,
                rotation,
            )
        entry = self._overlay_quad_entries.get(key)
        if (
            entry is None
            or entry["size"] != (width, height)
            or entry.get("format") != format_value
        ):
            if entry is not None:
                staging = entry.get("staging")
                if staging is not None:
                    staging.close()
                for resource in reversed(entry["resources"]):
                    self.vulkan.unregister_external_image(resource)
                self.xr.destroy_swapchain(entry["swapchain"])
            swapchain = self.xr.create_swapchain(
                self.session,
                self.xr.SwapchainCreateInfo(
                    usage_flags=(
                        self.xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT
                        | self.xr.SwapchainUsageFlags.TRANSFER_DST_BIT
                    ),
                    format=format_value,
                    sample_count=1,
                    width=width,
                    height=height,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                ),
            )
            images = list(
                self.xr.enumerate_swapchain_images(
                    swapchain, self.xr.SwapchainImageVulkan2KHR
                )
            )
            entry = {
                "swapchain": swapchain,
                "size": (width, height),
                "format": format_value,
                "resources": self._register_swapchain_images(
                    images, width, height, format_value
                ),
                "staging": None,
                "image_index": None,
                "content": None,
            }
            self._overlay_quad_entries[key] = entry
        if entry.get("content") is not request or entry.get("image_index") is None:
            with _acquired_swapchain_image(
                self.xr,
                _EyeSwapchain(
                    entry["swapchain"], [], width, height, entry["resources"]
                ),
            ) as image_index:
                rendered = renderer.render(
                    request, destination_format=int(entry["format"])
                )
                timeline = self.vulkan.copy_image(
                    rendered, entry["resources"][image_index]
                )
                renderer.notify_copy_timeline(timeline)
                entry["image_index"] = image_index
            entry["content"] = request
        image_index = int(entry["image_index"])
        if len(rotation) == 4:
            qx, qy, qz, qw = (float(value) for value in rotation)
        else:
            qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            layer_flags=(
                self.xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT
                | self.xr.CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT
            ),
            space=self.reference_space,
            eye_visibility=self.xr.EyeVisibility.BOTH,
            sub_image=self.xr.SwapchainSubImage(
                swapchain=entry["swapchain"],
                image_rect=self.xr.Rect2Di(
                    offset=self.xr.Offset2Di(x=0, y=0),
                    extent=self.xr.Extent2Di(width=width, height=height),
                ),
                image_array_index=0,
            ),
            pose=self.xr.Posef(
                orientation=self.xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=self.xr.Vector3f(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                ),
            ),
            size=self.xr.Extent2Df(width=float(size[0]), height=float(size[1])),
        )

    def _upload_tool_quad(self, key, rgba, position, size, rotation):
        if isinstance(rgba, VulkanMsdfQuadRequest):
            return self._upload_msdf_tool_quad(key, rgba, position, size, rotation)
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        entry = self._overlay_quad_entries.get(key)
        format_value = self._tool_quad_format()
        if (
            entry is None
            or entry["size"] != (width, height)
            or entry.get("format") != format_value
        ):
            if entry is not None:
                staging = entry.get("staging")
                if staging is not None:
                    staging.close()
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
                "format": format_value,
                "resources": self._register_swapchain_images(images, width, height, format_value),
                "staging": VulkanHostImage(self.vulkan, width, height, format=format_value, label=f"overlay-{key}"),
                "image_index": None,
                "content": None,
            }
            self._overlay_quad_entries[key] = entry
        if entry.get("content") is not rgba or entry.get("image_index") is None:
            entry["staging"].upload(rgba)
            with _acquired_swapchain_image(self.xr, _EyeSwapchain(entry["swapchain"], [], width, height, entry["resources"])) as image_index:
                self.vulkan.copy_image(entry["staging"].resource, entry["resources"][image_index])
                entry["image_index"] = image_index
            entry["content"] = rgba
        image_index = int(entry["image_index"])
        if len(rotation) == 4:
            qx, qy, qz, qw = (float(value) for value in rotation)
        else:
            qx, qy, qz, qw = _euler_degrees_to_quaternion(rotation)
        return self.xr.CompositionLayerQuad(
            layer_flags=(
                self.xr.CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT
                | self.xr.CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT
            ),
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

    def _tool_quad_format(self) -> int:
        if self._tool_quad_swapchain_format is None:
            formats = self.xr.enumerate_swapchain_formats(self.session)
            self._tool_quad_swapchain_format = _select_swapchain_format(
                self.vulkan.vk, list(formats), "srgb"
            )
        return int(self._tool_quad_swapchain_format)


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


def _update_filament_stereo_camera(
    bridge: Any,
    views: list[Any],
    *,
    near_plane: float = 0.05,
    far_plane: float = 1000.0,
) -> None:
    eye_models = [
        _xr_view_pose_to_model_mat4(view.pose) for view in views[:2]
    ]
    head_model = eye_models[0].copy()
    head_model[:3, 3] = 0.5 * (
        eye_models[0][:3, 3] + eye_models[1][:3, 3]
    )
    head_inverse = np.linalg.inv(head_model).astype(np.float32)
    position = tuple(float(value) for value in head_model[:3, 3])
    forward = head_model[:3, :3] @ (0.0, 0.0, -1.0)
    up = head_model[:3, :3] @ (0.0, 1.0, 0.0)
    center = tuple(position[index] + float(forward[index]) for index in range(3))
    bridge.set_camera_look_at(
        position, center, tuple(float(value) for value in up)
    )

    matrices: list[float] = []
    frustums: list[float] = []
    for view, eye_model in zip(views[:2], eye_models):
        matrices.extend(
            float(value)
            for value in (head_inverse @ eye_model).reshape(-1, order="F")
        )
        fov = view.fov
        frustums.extend(
            (
                math.tan(float(fov.angle_left)) * near_plane,
                math.tan(float(fov.angle_right)) * near_plane,
                math.tan(float(fov.angle_down)) * near_plane,
                math.tan(float(fov.angle_up)) * near_plane,
            )
        )
    bridge.set_stereo_camera(
        matrices,
        frustums,
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
    if mode not in {"srgb", "auto"}:
        raise ValueError(
            "OpenXR projection swapchain must use sRGB; "
            "linear UNORM output is not supported"
        )

    srgb = (
        vk.VK_FORMAT_R8G8B8A8_SRGB,
        vk.VK_FORMAT_B8G8R8A8_SRGB,
    )
    preferred = srgb
    for candidate in preferred:
        if int(candidate) in available_formats:
            return int(candidate)
    if available_formats:
        raise OpenXrVulkanUnavailableError(
            "OpenXR runtime exposes no sRGB projection swapchain format; "
            "refusing a color-space-changing UNORM fallback"
        )
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
