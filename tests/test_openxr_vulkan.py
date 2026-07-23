from __future__ import annotations

import ctypes
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vulkan as vk
import xr

from viewer.vulkan_context import (
    ImageState,
    ImageStateTracker,
    QueueFamilySelection,
    VulkanCapabilityError,
    VulkanContext,
    VulkanUnavailableError,
    _require_timeline_semaphore_features,
    format_vulkan_version,
    make_vulkan_version,
    unpack_vulkan_version,
    _find_queue_families,
)
from xr_viewer.core_openxr_vulkan import (
    OpenXrCompositionBuilder,
    OpenXrVulkanConfig,
    OpenXrVulkanPresenter,
    OpenXrVulkanUnavailableError,
    _EyeSwapchain,
    _scaled_dimension,
    _select_swapchain_format,
    _select_vulkan_api_version,
    _update_filament_camera,
)
from xr_viewer.controller_models import controller_button_local_position
from xr_viewer.overlay_textures import build_controller_callout_rgba
from xr_viewer.xr_math import _xr_quat_to_mat4


def test_vulkan_version_round_trip() -> None:
    packed = make_vulkan_version(1, 3, 275)
    assert unpack_vulkan_version(packed) == (1, 3, 275)
    assert format_vulkan_version(packed) == "1.3.275"


def test_timeline_feature_chain_returns_feature_node_and_sync_flag():
    class FeatureNode:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeVulkan:
        VK_TRUE = 1
        VK_FALSE = 0
        VkPhysicalDeviceTimelineSemaphoreFeatures = FeatureNode
        VkPhysicalDeviceSynchronization2Features = FeatureNode
        VkPhysicalDeviceFeatures2 = FeatureNode

        @staticmethod
        def vkGetPhysicalDeviceFeatures2(_physical_device, features2):
            features2.pNext.timelineSemaphore = 1
            features2.pNext.pNext.synchronization2 = 1

    feature_chain, synchronization2_enabled = _require_timeline_semaphore_features(
        FakeVulkan(), object()
    )

    assert feature_chain.synchronization2 == 1
    assert feature_chain.pNext.timelineSemaphore == 1
    assert synchronization2_enabled is True


def test_queue_family_selection_prefers_dedicated_compute_and_transfer() -> None:
    vk = SimpleNamespace(
        VK_QUEUE_GRAPHICS_BIT=0x1,
        VK_QUEUE_COMPUTE_BIT=0x2,
        VK_QUEUE_TRANSFER_BIT=0x4,
        vkGetPhysicalDeviceQueueFamilyProperties=lambda _device: [
            SimpleNamespace(queueCount=1, queueFlags=0x1 | 0x2),
            SimpleNamespace(queueCount=1, queueFlags=0x2),
            SimpleNamespace(queueCount=1, queueFlags=0x4),
        ],
    )
    assert _find_queue_families(vk, object()) == QueueFamilySelection(0, 1, 2)


def test_queue_family_selection_falls_back_to_graphics() -> None:
    vk = SimpleNamespace(
        VK_QUEUE_GRAPHICS_BIT=0x1,
        VK_QUEUE_COMPUTE_BIT=0x2,
        VK_QUEUE_TRANSFER_BIT=0x4,
        vkGetPhysicalDeviceQueueFamilyProperties=lambda _device: [
            SimpleNamespace(queueCount=1, queueFlags=0x1 | 0x2 | 0x4),
        ],
    )
    assert _find_queue_families(vk, object()) == QueueFamilySelection(0, 0, 0)


def test_image_state_tracker_returns_explicit_undefined_state() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    state = tracker.get(17, undefined_layout=9)
    assert state == ImageState(9, 0, 0, 3)


def test_image_state_tracker_updates_and_clears_state() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    state = ImageState(4, 8, 16, 2)
    tracker.update(17, state)
    assert tracker.get(17, undefined_layout=9) == state
    tracker.clear()
    assert tracker.get(17, undefined_layout=9).layout == 9


def test_image_state_tracker_removes_released_image() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    tracker.update(17, ImageState(4, 8, 16, 3))
    tracker.remove(17)
    assert tracker.get(17, undefined_layout=9) == ImageState(9, 0, 0, 3)


def test_image_state_tracker_rejects_wrong_queue_owner() -> None:
    tracker = ImageStateTracker(default_queue_family_index=3)
    tracker.update(17, ImageState(4, 8, 16, 2))
    with pytest.raises(VulkanCapabilityError, match="owned by queue family 2"):
        tracker.require_owner(17, 3)


def test_image_state_tracker_owns_pending_queue_transfer_until_acquire() -> None:
    tracker = ImageStateTracker(default_queue_family_index=0)
    tracker.update(17, ImageState(4, 8, 16, 0))
    transfer = tracker.begin_ownership_transfer(
        17,
        source_queue_family_index=0,
        destination_queue_family_index=2,
        undefined_layout=9,
    )
    with pytest.raises(VulkanCapabilityError, match="pending queue ownership"):
        tracker.require_owner(17, 2)
    tracker.complete_ownership_transfer(transfer)
    assert tracker.require_owner(17, 2).queue_family_index == 2


def test_openxr_version_range_clamps_requested_vulkan_version() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 2, 0),
        max_api_version_supported=xr.Version(1, 3, 0),
    )
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 0, 0)
    ) == make_vulkan_version(1, 2, 0)
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 2, 0)
    ) == make_vulkan_version(1, 2, 0)
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 4, 0)
    ) == make_vulkan_version(1, 3, 0)


def test_invalid_openxr_version_range_is_rejected() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 3, 0),
        max_api_version_supported=xr.Version(1, 1, 0),
    )
    with pytest.raises(OpenXrVulkanUnavailableError):
        _select_vulkan_api_version(requirements, make_vulkan_version(1, 2, 0))


def test_openxr_runtime_below_vulkan_12_is_rejected() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 0, 0),
        max_api_version_supported=xr.Version(1, 1, 0),
    )
    with pytest.raises(OpenXrVulkanUnavailableError, match="Vulkan 1.2"):
        _select_vulkan_api_version(requirements, make_vulkan_version(1, 4, 0))


def test_swapchain_format_prefers_srgb() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    assert _select_swapchain_format(vk, [44, 50, 43]) == 43
    with pytest.raises(OpenXrVulkanUnavailableError, match="no sRGB"):
        _select_swapchain_format(vk, [44])
    with pytest.raises(OpenXrVulkanUnavailableError):
        _select_swapchain_format(vk, [])


def test_swapchain_format_rejects_linear_unorm_mode() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    with pytest.raises(ValueError, match="must use sRGB"):
        _select_swapchain_format(vk, [43, 44], "unorm")
    assert _select_swapchain_format(vk, [43, 44], "srgb") == 43
    assert _select_swapchain_format(vk, [43, 44], "auto") == 43


def test_swapchain_color_mode_rejects_unknown_value() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    with pytest.raises(ValueError, match="must use sRGB"):
        _select_swapchain_format(vk, [43, 44], "linear")


def test_render_scale_is_bounded_by_runtime_limit() -> None:
    assert _scaled_dimension(1000, 1200, 0.5) == 500
    assert _scaled_dimension(1000, 1200, 2.0) == 1200
    assert _scaled_dimension(1, 1, 0.1) == 1


def test_presenter_validates_configuration() -> None:
    with pytest.raises(ValueError):
        OpenXrVulkanPresenter(OpenXrVulkanConfig(render_scale=0))


def test_openxr_defaults_to_validated_srgb_projection_target() -> None:
    assert OpenXrVulkanConfig().swapchain_color_mode == "srgb"
    assert OpenXrVulkanConfig().controller_model == "PICO"
    assert OpenXrVulkanConfig().controller_guide_max_distance == pytest.approx(0.4)


def test_presenter_rejects_non_positive_controller_guide_distance() -> None:
    with pytest.raises(ValueError, match="controller_guide_max_distance"):
        OpenXrVulkanPresenter(OpenXrVulkanConfig(controller_guide_max_distance=0.0))


def test_controller_callout_texture_keeps_controller_center_transparent() -> None:
    rgba = build_controller_callout_rgba(lang="CN")

    assert rgba.shape == (1536, 2048, 4)
    assert rgba.dtype == np.uint8
    assert rgba[768, 1024, 3] == 0
    assert tuple(rgba[768, 1024, :3]) == (255, 255, 255)
    assert tuple(rgba[420, 1500]) == (255, 255, 255, 255)
    assert rgba[420, 1200, 3] == 0
    assert tuple(rgba[600, 1080]) == (255, 255, 255, 255)
    assert rgba[252, 300, 3] == 0
    assert int(rgba[..., 3].max()) == 255


def test_controller_guide_pose_hides_beyond_headset_distance() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._grip_mat_r = np.eye(4, dtype=np.float64)
    presenter._aim_mat_r = np.eye(4, dtype=np.float64)
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.4), dtype=np.float64)

    pose = presenter._controller_guide_pose()
    assert pose is not None
    assert pose[1] == pytest.approx((0.34, 0.255))
    assert np.linalg.norm(np.asarray(pose[2], dtype=np.float64)) == pytest.approx(1.0)

    presenter._head_position_w[2] = 0.401
    assert presenter._controller_guide_pose() is None


@pytest.mark.parametrize(
    ("brand_name", "expected"),
    (
        ("HP", (-0.0235, 0.012129, -0.035076)),
        ("INDEX", (-0.021801, -0.001037, -0.051047)),
        ("PICO", (-0.00672205, 0.01771696, -0.02744452)),
        ("QUEST", (-0.0128, 0.001141, -0.028491)),
        ("VIVE", (-0.021922, 0.00029, -0.041995)),
        ("YVR", (-0.022195, 0.008466, -0.007238)),
    ),
)
def test_b_button_position_is_resolved_from_each_controller_glb(
    brand_name: str, expected: tuple[float, float, float]
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / f"src/xr_viewer/controllers/{brand_name}/right.glb"
    )

    position = controller_button_local_position(str(path), "b_button")

    assert position == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("brand_name", "expected_multiplier"),
    (
        ("HP", 1.5),
        ("INDEX", 1.5),
        ("PICO", 1.0),
        ("QUEST", 1.0),
        ("VIVE", 1.5),
        ("YVR", 1.5),
    ),
)
def test_controller_profile_selects_ambient_light_multiplier(
    brand_name: str, expected_multiplier: float
) -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._controller_brand = presenter._controller_brands[brand_name]
    presenter._filament_ambient_light_color = (0.06, 0.05, 0.05)

    assert presenter._controller_brand.ambient_light_multiplier == pytest.approx(
        expected_multiplier
    )
    assert presenter._controller_ambient_light_color() == pytest.approx(
        tuple(value * expected_multiplier for value in (0.06, 0.05, 0.05))
    )


def test_controller_brand_switch_recalculates_b_button_anchor() -> None:
    presenter = OpenXrVulkanPresenter()
    previous_brand = presenter._controller_brand
    presenter._controller_b_button_local = np.asarray(
        (99.0, 99.0, 99.0), dtype=np.float64
    )
    presenter._controller_b_button_resolved = True

    presenter._switch_shortcut_controller_brand()

    assert presenter._controller_brand is not previous_brand
    expected = controller_button_local_position(
        str(presenter._controller_brand.right_glb), "b_button"
    )
    assert presenter._controller_b_button_resolved is True
    assert presenter._controller_b_button_local == pytest.approx(expected)


def test_controller_brand_switch_refreshes_ambient_light() -> None:
    class Bridge:
        def __init__(self) -> None:
            self.ambient_colors: list[tuple[float, float, float]] = []

        def load_controller(self, _hand: int, _data: bytes) -> None:
            pass

        def set_ambient_light(self, color) -> None:
            self.ambient_colors.append(tuple(color))

    presenter = OpenXrVulkanPresenter()
    presenter._filament_ambient_light_color = (0.06, 0.05, 0.05)
    presenter._controller_brand = presenter._controller_brands["QUEST"]
    presenter.filament_bridge = Bridge()

    presenter._switch_shortcut_controller_brand()

    assert presenter._controller_brand.name == "VIVE"
    assert len(presenter.filament_bridge.ambient_colors) == 1
    assert presenter.filament_bridge.ambient_colors[0] == pytest.approx(
        (0.09, 0.075, 0.075)
    )


def test_controller_button_position_does_not_require_opengl_renderer(
    monkeypatch,
) -> None:
    import builtins

    path = (Path(__file__).resolve().parents[1] /
            "src/xr_viewer/controllers/PICO/right.glb")
    original_import = builtins.__import__

    def reject_moderngl(name, *args, **kwargs):
        if name == "moderngl":
            raise ModuleNotFoundError("moderngl intentionally unavailable")
        return original_import(name, *args, **kwargs)

    controller_button_local_position.cache_clear()
    monkeypatch.setattr(builtins, "__import__", reject_moderngl)

    assert controller_button_local_position(str(path), "b_button") is not None


def test_controller_guide_stays_head_facing_while_endpoint_follows_b_button() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.2), dtype=np.float64)
    presenter._grip_mat_r = np.eye(4, dtype=np.float64)
    presenter._aim_mat_r = np.eye(4, dtype=np.float64)
    presenter._controller_b_button_local = np.asarray(
        (-0.00672205, 0.01771696, -0.02744452), dtype=np.float64
    )
    presenter._controller_b_button_resolved = True

    def endpoint_and_facing():
        position, size, quaternion = presenter._controller_guide_pose()
        orientation = xr.Quaternionf(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )
        basis = _xr_quat_to_mat4(orientation)[:3, :3]
        endpoint_local = np.asarray((
            (540.0 / 1024.0 - 0.5) * size[0],
            (0.5 - 300.0 / 768.0) * size[1],
            0.0,
        ))
        endpoint = np.asarray(position) + basis @ endpoint_local
        button = presenter._controller_b_button_world_position()
        toward_head = presenter._head_position_w - np.asarray(position)
        toward_head /= np.linalg.norm(toward_head)
        return endpoint, button, float(np.dot(basis[:, 2], toward_head))

    initial_endpoint, initial_button, initial_facing = endpoint_and_facing()
    assert np.linalg.norm(initial_endpoint - initial_button) == pytest.approx(0.006, abs=1e-5)
    assert initial_facing > 0.99

    angle = math.radians(30.0)
    rotation = np.asarray((
        (math.cos(angle), -math.sin(angle), 0.0),
        (math.sin(angle), math.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
    presenter._grip_mat_r[:3, :3] = rotation
    presenter._aim_mat_r[:3, :3] = rotation
    rotated_endpoint, rotated_button, rotated_facing = endpoint_and_facing()

    assert not np.allclose(initial_button, rotated_button)
    assert np.linalg.norm(rotated_endpoint - rotated_button) == pytest.approx(0.006, abs=1e-5)
    assert rotated_facing > 0.99


def test_controller_callout_uses_projection_layer_not_quad_layer() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "bridge.set_controller_guide_texture(self._controller_callout_rgba)" in source
    assert "bridge.set_controller_guide(guide_matrix, visible=True)" in source
    assert 'specs.append(("controller_callouts"' not in source
    assert 'if self._operation_guide_visible:\n            rgba = build_help_rgba' in source


def test_filament_controller_guide_tracks_geometry_and_visibility() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = np.asarray((0.0, 0.0, 0.2), dtype=np.float64)
    presenter._grip_mat_r = np.eye(4, dtype=np.float64)
    presenter._controller_b_button_local = np.asarray(
        (-0.00672205, 0.01771696, -0.02744452), dtype=np.float64
    )
    presenter._controller_b_button_resolved = True

    class Bridge:
        controller_guide_abi_available = True

        def __init__(self):
            self.calls = []

        def set_controller_guide(self, matrix, *, visible):
            self.calls.append((np.asarray(matrix).copy(), visible))

    bridge = Bridge()
    presenter._update_filament_controller_guide(bridge)

    matrix, visible = bridge.calls[-1]
    assert visible is True
    assert matrix.shape == (4, 4)
    assert np.linalg.norm(matrix[:3, 0]) == pytest.approx(0.34)
    assert np.linalg.norm(matrix[:3, 1]) == pytest.approx(0.255)
    assert np.dot(matrix[:3, 2], presenter._head_position_w - matrix[:3, 3]) > 0.0

    presenter._head_position_w[2] = 0.401
    presenter._update_filament_controller_guide(bridge)
    _, visible = bridge.calls[-1]
    assert visible is False


def test_presenter_defaults_to_capability_gated_zero_copy_path(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_FILAMENT_SCREEN_IMAGE", raising=False)
    presenter = OpenXrVulkanPresenter()
    assert presenter._filament_screen_image_enabled is True


def test_filament_screen_image_requires_per_eye_external_ready_semaphores() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen_image_enabled = True

    class Bridge:
        screen_image_abi_available = True
        screen_ready_semaphore_abi_available = True

    presenter.filament_bridge = Bridge()
    frame = SimpleNamespace(
        metadata={
            "vulkan_output_sync": "cuda_stream_synchronized",
            "vulkan_ready_semaphore_left": object(),
            "vulkan_ready_semaphore_right": object(),
        }
    )
    assert presenter._can_use_filament_screen_image(frame) is False
    frame.metadata["vulkan_output_sync"] = "cuda_external_semaphore"
    assert presenter._can_use_filament_screen_image(frame) is True


def test_host_image_upload_writes_padded_rows_without_pointer_cast() -> None:
    from viewer.vulkan_resources import VulkanHostImage
    import numpy as np

    image = VulkanHostImage.__new__(VulkanHostImage)
    image.width = 2
    image.height = 2
    image._layout = SimpleNamespace(offset=2, rowPitch=12, size=26)
    mapped = bytearray(26)
    image.vk = SimpleNamespace(
        vkMapMemory=lambda *args: mapped,
        vkUnmapMemory=lambda *args: None,
    )
    image.context = SimpleNamespace(device=object())
    image.memory = object()
    image.upload(np.arange(16, dtype=np.uint8).reshape(2, 2, 4))
    assert mapped[2:10] == bytes(range(8))
    assert mapped[14:22] == bytes(range(8, 16))


def test_presenter_uses_controller_action_mixin_initializer() -> None:
    presenter = OpenXrVulkanPresenter()
    assert hasattr(presenter, "_init_controller_actions")
    assert not hasattr(presenter, "_initialize_controller_actions")
    assert presenter._LASER_HIDE_AFTER == 5.0
    assert presenter._laser_prev_mat_l is None
    assert presenter._laser_prev_mat_r is None


def test_filament_controller_lifecycle_hides_each_idle_hand_independently() -> None:
    class Bridge:
        controller_abi_available = True
        controller_visibility_abi_available = True
        laser_abi_available = True

        def __init__(self) -> None:
            self.visible = []
            self.poses = []
            self.inputs = []
            self.lasers = []

        def set_controller_visible(self, hand, visible) -> None:
            self.visible.append((hand, visible))

        def set_controller_pose(self, hand, matrix) -> None:
            self.poses.append((hand, matrix.copy()))

        def set_controller_inputs(self, hand, **values) -> None:
            self.inputs.append((hand, values))

        def set_controller_laser(self, hand, matrix, *, visible) -> None:
            self.lasers.append((hand, matrix.copy(), visible))

    presenter = OpenXrVulkanPresenter()
    presenter._controller_brand = SimpleNamespace(
        offset=(0.0, 0.0, 0.0), rotation_deg=0.0
    )
    presenter._frame_now = 20.0
    presenter._laser_last_move_l = 14.9
    presenter._laser_last_move_r = 19.0
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._aim_mat_l = None
    presenter._aim_mat_r = None
    presenter._controller_inputs = ({}, {"joystick_touched": 1.0})
    bridge = Bridge()

    presenter._update_filament_controllers(bridge)

    assert bridge.visible == [(0, False), (1, True)]
    assert [hand for hand, _matrix in bridge.poses] == [1]
    assert [hand for hand, _values in bridge.inputs] == [1]
    assert bridge.inputs[0][1]["button_mask"] == 1 << 6
    assert [(hand, visible) for hand, _matrix, visible in bridge.lasers] == [
        (0, False), (1, False)
    ]


def test_controller_touch_actions_cover_thumbstick_trackpad_and_thumbrest() -> None:
    root = Path(__file__).resolve().parents[1]
    actions = (root / "src/xr_viewer/core_controller_actions.py").read_text(
        encoding="utf-8"
    )
    inputs = (root / "src/xr_viewer/core_controller_input.py").read_text(
        encoding="utf-8"
    )

    assert "/input/thumbstick/touch" in actions
    assert "/input/trackpad/touch" in actions
    assert "/input/thumbrest/touch" in actions
    assert 'left["stick_click"]' in inputs
    assert '"joystick_touched": 1.0 if left_touched else 0.0' in inputs
    assert '"touchpad_touched": 1.0 if right_touched else 0.0' in inputs


def test_active_filament_controller_uses_legacy_laser_calibration() -> None:
    class Bridge:
        controller_abi_available = True
        controller_visibility_abi_available = True
        laser_abi_available = True

        def __init__(self) -> None:
            self.laser_matrix = None

        def set_controller_visible(self, hand, visible) -> None:
            pass

        def set_controller_pose(self, hand, matrix) -> None:
            pass

        def set_controller_inputs(self, hand, **values) -> None:
            pass

        def set_controller_laser(self, hand, matrix, *, visible) -> None:
            if hand == 0 and visible:
                self.laser_matrix = matrix.copy()

    presenter = OpenXrVulkanPresenter()
    presenter._controller_brand = SimpleNamespace(
        offset=(0.0, 0.0, 0.0), rotation_deg=0.0
    )
    presenter._frame_now = 20.0
    presenter._laser_last_move_l = 19.0
    presenter._laser_last_move_r = 0.0
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = None
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._aim_mat_r = None
    presenter._controller_inputs = ({}, {})
    bridge = Bridge()

    presenter._update_filament_controllers(bridge)

    assert bridge.laser_matrix is not None
    assert np.linalg.norm(bridge.laser_matrix[:3, 0]) == pytest.approx(0.006)
    assert np.linalg.norm(bridge.laser_matrix[:3, 1]) == pytest.approx(0.4)
    assert np.linalg.norm(bridge.laser_matrix[:3, 2]) == pytest.approx(0.006)
    assert bridge.laser_matrix[1, 3] > 0.0
    assert bridge.laser_matrix[2, 3] < 0.0


def test_vulkan_presenter_exposes_legacy_overlay_shortcut_state() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._controller_inputs = (
        {"x_button": 0.0, "menu_button": 0.0},
        {"a_button": 0.0, "b_button": 0.0, "menu_button": 0.0},
    )
    presenter._handle_controller_shortcuts()
    presenter._controller_inputs = (
        {"x_button": 1.0, "menu_button": 0.0},
        {"a_button": 0.0, "b_button": 0.0, "menu_button": 0.0},
    )
    presenter._handle_controller_shortcuts()
    presenter._controller_inputs = ({"x_button": 0.0}, {})
    presenter._handle_controller_shortcuts()
    assert presenter._keyboard_visible is True
    presenter._controller_inputs = ({"x_button": 1.0}, {})
    presenter._handle_controller_shortcuts()
    presenter._controller_inputs = ({"x_button": 0.0}, {})
    presenter._handle_controller_shortcuts()
    assert presenter._keyboard_visible is False


def test_vulkan_b_long_press_displays_operation_guide() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._frame_now = 1.0
    presenter._controller_inputs = ({}, {"b_button": 1.0})
    presenter._handle_controller_shortcuts()

    presenter._frame_now = 2.01
    presenter._handle_controller_shortcuts()

    assert presenter._operation_guide_visible is True
    assert presenter._fps_overlay_visible is False
    assert presenter._aperture_visible is False


def test_vulkan_shortcuts_cycle_screen_preset_and_background() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._head_position_w = (0.0, 0.0, 0.0)
    presenter._filament_screen = (
        (0.0, 0.0, -16.0),
        16.0,
        9.0,
        (0.0, 0.0, 0.0),
    )

    presenter._dispatch_controller_shortcut("cycle_screen_preset")

    position, width, height, _rotation = presenter._filament_screen
    assert position == pytest.approx((0.0, 0.0, -20.0))
    assert width == pytest.approx(22.0)
    assert height == pytest.approx(12.375)

    presenter._dispatch_controller_shortcut("toggle_background")
    assert presenter._filament_skybox_brightness == pytest.approx(0.0)
    presenter._dispatch_controller_shortcut("toggle_background")
    assert presenter._filament_skybox_brightness == pytest.approx(1.0)


def test_vulkan_reset_screen_restores_initial_size_and_pose() -> None:
    presenter = OpenXrVulkanPresenter()
    initial = ((0.0, 0.0, -2.5), 2.4, 1.35, (0.0, 0.0, 0.0))
    presenter._filament_screen_initial = initial
    presenter._filament_screen = (
        (1.0, 0.5, -20.0), 22.0, 12.375, (5.0, 10.0, 0.0)
    )

    presenter._dispatch_controller_shortcut("reset_screen")

    assert presenter._filament_screen == initial


def test_right_grip_moves_screen_without_resizing_it() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)
    presenter._grip_mat_r = np.eye(4, dtype=np.float32)
    presenter._controller_inputs = ({}, {"grip": 1.0})

    presenter._handle_vulkan_pointer_input()
    presenter._grip_mat_r[0, 3] = 0.2
    presenter._handle_vulkan_pointer_input()

    position, width, height, _rotation = presenter._filament_screen
    assert position == pytest.approx((0.2, 0.0, -2.0))
    assert width == pytest.approx(2.4)
    assert height == pytest.approx(1.35)


def test_left_grip_rotation_snaps_screen_to_quarter_turn() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._grip_mat_l = np.eye(4, dtype=np.float32)
    presenter._grip_rotation_anchor_l = np.eye(3, dtype=np.float64)
    presenter._screen_rotation_anchor_l = (0.0, 0.0, 0.0)
    angle = math.radians(100.0)
    presenter._grip_mat_l[:3, :3] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )

    presenter._apply_grip_screen_rotation(0)

    assert presenter._filament_screen[3] == pytest.approx((0.0, 0.0, 90.0))


def test_keyboard_world_position_is_converted_to_screen_relative_offset() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (1.0, 2.0, -3.0), 2.4, 1.0, (0.0, 0.0, 0.0)
    )

    presenter._set_keyboard_world_position((1.5, 1.0, -2.5))

    assert presenter._keyboard_pose_mat4()[:3, 3] == pytest.approx(
        (1.5, 1.0, -2.5)
    )


def test_continuous_screen_shortcuts_apply_only_while_laser_hits_screen() -> None:
    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter._head_position_w = (0.0, 0.0, 0.0)
    presenter._aim_mat_l = np.eye(4, dtype=np.float32)
    presenter._aim_mat_r = np.eye(4, dtype=np.float32)

    presenter._dispatch_controller_shortcut(
        "rotate_screen", yaw_delta=10.0, pitch_delta=5.0
    )
    presenter._dispatch_controller_shortcut(
        "resize_screen", width_delta=0.6, distance_delta=0.5
    )

    position, width, _height, rotation = presenter._filament_screen
    assert rotation == pytest.approx((10.0, 5.0, 0.0))
    assert width == pytest.approx(3.0)
    assert np.linalg.norm(np.asarray(position)) == pytest.approx(2.5)


def test_controller_brand_switch_and_calibration_save_use_live_profile(
    tmp_path,
) -> None:
    class Bridge:
        def __init__(self) -> None:
            self.loaded: list[tuple[int, bytes]] = []

        def load_controller(self, hand: int, data: bytes) -> None:
            self.loaded.append((hand, data))

    def brand(name: str, marker: bytes):
        root = tmp_path / name
        root.mkdir()
        left = root / "left.glb"
        right = root / "right.glb"
        left.write_bytes(marker + b"L")
        right.write_bytes(marker + b"R")
        (root / "profile.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            name=name,
            root=root,
            left_glb=left,
            right_glb=right,
            offset=(0.0, 0.0, 0.0),
            rotation_deg=0.0,
        )

    first = brand("A", b"A")
    second = brand("B", b"B")
    presenter = OpenXrVulkanPresenter()
    presenter._controller_brands = {"A": first, "B": second}
    presenter._controller_brand = first
    presenter.filament_bridge = Bridge()

    presenter._dispatch_controller_shortcut("switch_controller_brand")
    presenter._controller_calibration_offset[:] = (0.1, 0.2, 0.3)
    presenter._controller_calibration_rotation_deg = 12.5
    presenter._controller_calibration_mode = True
    presenter._dispatch_controller_shortcut("save_controller_calibration")

    profile = json.loads((second.root / "profile.json").read_text(encoding="utf-8"))
    assert presenter._controller_brand is second
    assert presenter.filament_bridge.loaded == [(0, b"BL"), (1, b"BR")]
    assert profile["overrides"] == {
        "model_offset": [0.1, 0.2, 0.3],
        "model_rotation_deg": 12.5,
    }
    assert presenter._controller_calibration_mode is False


def test_vulkan_shortcut_delegates_runtime_owned_actions() -> None:
    actions: list[str] = []
    presenter = OpenXrVulkanPresenter(
        on_controller_shortcut=lambda action: actions.append(action) or True
    )

    presenter._dispatch_controller_shortcut("toggle_stereo")
    presenter._dispatch_controller_shortcut("reset_depth")

    assert actions == ["toggle_stereo", "reset_depth"]
    assert presenter._unsupported_shortcut_actions == set()


def test_vulkan_shortcut_toggles_native_curved_screen() -> None:
    class Bridge:
        screen_curved_abi_available = True

        def __init__(self) -> None:
            self.curved: list[bool] = []
            self.screens: list[tuple] = []

        def set_screen_curved(self, curved: bool) -> None:
            self.curved.append(curved)

        def set_screen(self, *screen) -> None:
            self.screens.append(screen)

    presenter = OpenXrVulkanPresenter()
    presenter._filament_screen = (
        (0.0, 0.0, -2.0), 2.4, 1.35, (0.0, 0.0, 0.0)
    )
    presenter.filament_bridge = Bridge()

    presenter._dispatch_controller_shortcut("toggle_screen_shape")
    presenter._dispatch_controller_shortcut("toggle_screen_shape")

    assert presenter.filament_bridge.curved == [True, False]
    assert len(presenter.filament_bridge.screens) == 2


def test_vulkan_shortcut_toggles_legacy_green_passthrough_backdrop() -> None:
    class Bridge:
        passthrough_backdrop_abi_available = True

        def __init__(self) -> None:
            self.values: list[bool] = []

        def set_passthrough_backdrop(self, enabled: bool) -> None:
            self.values.append(enabled)

    presenter = OpenXrVulkanPresenter()
    presenter.filament_bridge = Bridge()

    presenter._dispatch_controller_shortcut("toggle_passthrough")
    presenter._dispatch_controller_shortcut("toggle_passthrough")

    assert presenter.filament_bridge.values == [True, False]


def test_openxr_frame_gate_waits_for_runtime_output_before_filament() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "if self._pending_output is None and not self._has_presented_frame:" in source
    assert "waiting for first runtime eye frame" in source
    assert "layer = self._render_projection_layer(views, output_frame)" in source
    assert "bridge.set_screen_image(" in source
    assert "D2S_ENABLE_FILAMENT_SCREEN_IMAGE" in source


def test_quad_layer_uses_runtime_output_size_and_openxr_visibility() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "_ensure_quad_swapchains(width, height)" in source
    assert '_select_swapchain_format(vk, formats, "srgb")' in source
    assert "flip_x=True" not in source
    assert "flip_y=False" in source


def test_tool_quad_layer_enables_unpremultiplied_source_alpha() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "CompositionLayerFlags.BLEND_TEXTURE_SOURCE_ALPHA_BIT" in source
    assert "CompositionLayerFlags.UNPREMULTIPLIED_ALPHA_BIT" in source
    assert "format_value if format_value is not None" in source
    assert "CompositionLayerQuad" in source
    assert "EyeVisibility.LEFT" in source


def test_profile_reference_space_is_shared_with_controller_pose_queries() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "self.reference_space = new_space" in source
    assert "self._xr_space = new_space" in source


def test_profile_screen_height_defaults_to_16_9_width() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert 'float(screen.get("width", 2.4)) * 9.0 / 16.0' in source
    assert "EyeVisibility.RIGHT" in source
    assert "_has_presented_frame" in source
    assert "self._last_quad_layers" in source
    assert "Render the world at the current headset pose" in source


def test_vulkan_copy_allows_srgb_unorm_quad_conversion() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/viewer/vulkan_context.py").read_text(encoding="utf-8")

    assert "formats_are_srgb_compatible" in source
    assert "or not formats_match" in source


def test_profile_pose_is_applied_once_to_openxr_reference_space() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "_apply_profile_reference_space(views)" in source
    assert "self._profile_space_applied = True" in source
    assert "reference_head = self._level_head_model_mat4(raw_head)" in source
    assert "space_pose = reference_head @ np.linalg.inv(self._profile_head_transform)" in source
    assert "xr.ReferenceSpaceType.STAGE" in source
    assert "enumerate_reference_spaces(self.session)" in source


def test_filament_profile_keeps_glb_and_screen_positions_separate(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "model_position": [0.0, 0.0, 0.0],
        "view_poses": [{"x": 1.0, "y": 2.0, "z": 3.0}],
        "screen": {"position": [10.0, 20.0, 30.0]},
    }), encoding="utf-8")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_profile_path=str(profile_path),
    ))
    presenter._load_filament_profile()
    assert presenter._profile_head_transform[:3, 3].tolist() == [1.0, 2.0, 3.0]
    assert presenter._filament_screen[0] == (10.0, 20.0, 30.0)


def test_filament_profile_view_pose_is_converted_to_glb_local_space(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "model_position": [0.0, 843.0, 0.0],
        "view_poses": [{"x": -24.0, "y": 900.0, "z": -961.0}],
    }), encoding="utf-8")
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        filament_profile_path=str(profile_path),
    ))
    presenter._load_filament_profile()
    assert presenter._profile_head_transform[:3, 3].tolist() == [-24.0, 57.0, -961.0]


def test_controller_profile_rotation_uses_local_x_axis() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")
    assert (
        "0.0, math.radians(self._controller_calibration_rotation_deg), 0.0"
        in source
    )


def test_quad_profile_rotation_uses_legacy_yaw_pitch_roll_order() -> None:
    from xr_viewer.core_openxr_vulkan import _euler_degrees_to_quaternion

    x, y, z, w = _euler_degrees_to_quaternion((90.0, 0.0, 0.0))
    assert abs(x) < 1e-6
    assert abs(y - 2 ** -0.5) < 1e-6
    assert abs(z) < 1e-6
    assert abs(w - 2 ** -0.5) < 1e-6


def test_projection_layer_binds_matching_runtime_eye_to_filament_screen() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")
    assert "bridge.set_screen_image(" in source
    assert "output_frame.left_eye" in source
    assert "output_frame.right_eye" in source
    assert "screen_image_abi_available" in source
    assert "self._filament_screen_image_enabled" in source
    assert "The main virtual screen is rendered in the Projection Layer" in source


def test_presenter_run_until_owns_shutdown_close() -> None:
    presenter = OpenXrVulkanPresenter()
    shutdown = threading.Event()
    calls = []

    presenter.initialize = lambda: calls.append("initialize")
    presenter.run_frame = lambda: (calls.append("frame"), shutdown.set(), True)[2]
    presenter.close = lambda: calls.append("close")

    assert presenter.run_until(shutdown) == 0
    assert calls == ["initialize", "frame", "close"]


def test_presenter_waits_for_headset_and_retries_initialization(capsys) -> None:
    presenter = OpenXrVulkanPresenter(OpenXrVulkanConfig(
        openxr_no_headset_retry_interval=0.001,
        openxr_standby_retry_interval=0.001,
        openxr_standby_retry_max_interval=0.001,
    ))
    shutdown = threading.Event()
    calls = []

    def initialize():
        calls.append("initialize")
        if calls.count("initialize") == 1:
            raise type("FormFactorUnavailableError", (RuntimeError,), {})()

    def run_frame():
        calls.append("frame")
        shutdown.set()
        return True

    presenter.initialize = initialize
    presenter.run_frame = run_frame
    presenter.close = lambda: calls.append("close")

    assert presenter.run_until(shutdown) == 0
    assert calls == ["initialize", "close", "initialize", "frame", "close"]
    assert "Vulkan/Filament initialization deferred" in capsys.readouterr().out


def test_presenter_wait_enters_hard_idle_after_configured_timeout(capsys) -> None:
    states = []
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(headset_wait_inference_timeout=0.0),
        on_headset_state=states.append,
    )

    presenter._notify_headset_waiting()
    assert states == ["waiting", "hard_idle"]
    assert "Headset not detected or in standby" in capsys.readouterr().out

    presenter._notify_headset_active()
    assert states[-1] == "active"


def test_presenter_rejects_output_while_headset_is_waiting() -> None:
    from types import SimpleNamespace

    presenter = OpenXrVulkanPresenter()

    with pytest.raises(RuntimeError, match="waiting for headset rendering"):
        presenter.submit_output(object())

    presenter._notify_headset_active()
    presenter.session_running = True
    with pytest.raises(TypeError, match="VulkanImageResource"):
        presenter.submit_output(SimpleNamespace(left_eye=None, right_eye=None))


def test_filament_bridge_binds_each_openxr_eye(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeBridge:
        def __init__(self, path):
            calls.append(("load", path))

        def create(self, **kwargs):
            calls.append(("create", kwargs["device"]))

        def create_eye_swapchain(self, eye_index, images, **kwargs):
            calls.append(("swapchain", (eye_index, list(images), kwargs["format"])))

        def set_scene_exposure(self, _value):
            pass

        def set_skybox_brightness(self, _value):
            pass

        def set_fill_light(self, _color, _intensity, _direction):
            pass

        def close(self):
            calls.append(("close", None))

    import xr_viewer.filament_vulkan_bridge as bridge_module

    monkeypatch.setattr(bridge_module, "FilamentVulkanBridge", FakeBridge)
    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(filament_bridge_path="bridge.dll")
    )
    presenter.vulkan = SimpleNamespace(
        instance=1,
        physical_device=2,
        device=3,
        queue_family_index=4,
    )
    presenter.swapchain_format = 43
    presenter.swapchains = [
        _EyeSwapchain("left", [SimpleNamespace(image="left-image")], 10, 20),
        _EyeSwapchain("right", [SimpleNamespace(image="right-image")], 30, 40),
    ]

    presenter._initialize_filament_bridges()

    assert presenter.filament_bridge is not None
    assert calls == [
        ("load", "bridge.dll"),
        ("create", 3),
        ("swapchain", (0, ["left-image"], 43)),
        ("swapchain", (1, ["right-image"], 43)),
    ]


def test_filament_camera_receives_openxr_pose_and_fov() -> None:
    calls: list[tuple[str, tuple[float, ...]]] = []

    class FakeBridge:
        def set_camera_look_at(self, eye, center, up):
            calls.append(("look_at", (*eye, *center, *up)))

        def set_camera_projection(self, fov_degrees, aspect, **kwargs):
            calls.append(("projection", (fov_degrees, aspect, kwargs)))

    view = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        fov=SimpleNamespace(
            angle_left=-0.7,
            angle_right=0.7,
            angle_up=0.6,
            angle_down=-0.6,
        ),
    )

    _update_filament_camera(FakeBridge(), view)

    assert calls[0][0] == "look_at"
    assert calls[0][1][:3] == (1.0, 2.0, 3.0)
    assert calls[0][1][3:6] == (1.0, 2.0, 2.0)
    assert calls[0][1][6:] == (0.0, 1.0, 0.0)
    assert calls[1][0] == "projection"
    assert calls[1][1][0] == pytest.approx(68.7549, rel=1e-4)
    assert calls[1][1][2]["far_plane"] == 1000.0


def test_swapchain_image_is_released_when_wait_fails() -> None:
    calls: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            calls.append("acquire")
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            calls.append("wait")
            raise RuntimeError("wait failed")

        @staticmethod
        def release_swapchain_image(_handle):
            calls.append("release")

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.swapchains = [
        _EyeSwapchain(
            handle=object(),
            images=[SimpleNamespace(image=None)],
            width=1,
            height=1,
        )
    ]
    with pytest.raises(RuntimeError, match="wait failed"):
        presenter._render_projection_layer([object()])
    assert calls == ["acquire", "wait", "release"]


def test_swapchain_image_is_released_after_wait_when_render_fails() -> None:
    calls: list[str] = []

    class FakeXr:
        INFINITE_DURATION = 1

        @staticmethod
        def acquire_swapchain_image(_handle):
            calls.append("acquire")
            return 0

        @staticmethod
        def wait_swapchain_image(_handle, _wait_info):
            calls.append("wait")

        @staticmethod
        def release_swapchain_image(_handle):
            calls.append("release")

        @staticmethod
        def SwapchainImageWaitInfo(*, timeout):
            return timeout

    class FakeVulkan:
        @staticmethod
        def image_handle_from_address(_address):
            return object()

        @staticmethod
        def clear_color_image(_image, _color):
            raise RuntimeError("clear failed")

    presenter = OpenXrVulkanPresenter()
    presenter.xr = FakeXr
    presenter.vulkan = FakeVulkan()
    presenter.swapchains = [
        _EyeSwapchain(
            handle=object(),
            images=[SimpleNamespace(image=ctypes.c_void_p(1))],
            width=1,
            height=1,
        )
    ]
    with pytest.raises(RuntimeError, match="clear failed"):
        presenter._render_projection_layer([object()])
    assert calls == ["acquire", "wait", "release"]


def test_projection_layer_builder_owns_only_layer_assembly() -> None:
    class FakeXr:
        CompositionLayerProjectionView = staticmethod(lambda **kwargs: kwargs)
        SwapchainSubImage = staticmethod(lambda **kwargs: kwargs)
        Rect2Di = staticmethod(lambda **kwargs: kwargs)
        Offset2Di = staticmethod(lambda **kwargs: kwargs)
        Extent2Di = staticmethod(lambda **kwargs: kwargs)
        CompositionLayerProjection = staticmethod(lambda **kwargs: kwargs)

    views = [
        SimpleNamespace(pose="left-pose", fov="left-fov"),
        SimpleNamespace(pose="right-pose", fov="right-fov"),
    ]
    swapchains = [
        _EyeSwapchain("left-chain", [], 10, 20),
        _EyeSwapchain("right-chain", [], 30, 40),
    ]
    layer = OpenXrCompositionBuilder(FakeXr, "local-space").projection_layer(
        views, swapchains
    )
    assert layer["space"] == "local-space"
    assert [view["pose"] for view in layer["views"]] == ["left-pose", "right-pose"]
    assert layer["views"][1]["sub_image"]["image_rect"]["extent"] == {
        "width": 30,
        "height": 40,
    }


def test_standalone_vulkan_context_smoke() -> None:
    try:
        context = VulkanContext.create()
    except (
        VulkanUnavailableError,
        VulkanCapabilityError,
        vk.VkErrorIncompatibleDriver,
    ) as exc:
        pytest.skip(str(exc))
    try:
        assert context.device_info.name
        assert context.device_info.queue_family_index >= 0
    finally:
        context.close()
    assert context.closed
