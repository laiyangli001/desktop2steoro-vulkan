from __future__ import annotations

import ctypes
import json
import threading
from pathlib import Path
from types import SimpleNamespace

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
    assert _select_swapchain_format(vk, [44]) == 44
    with pytest.raises(OpenXrVulkanUnavailableError):
        _select_swapchain_format(vk, [])


def test_swapchain_format_unorm_mode_prefers_unorm_for_ab_test() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    assert _select_swapchain_format(vk, [43, 44], "unorm") == 44
    assert _select_swapchain_format(vk, [43, 44], "srgb") == 43
    assert _select_swapchain_format(vk, [43, 44], "auto") == 43


def test_swapchain_color_mode_rejects_unknown_value() -> None:
    vk = SimpleNamespace(
        VK_FORMAT_R8G8B8A8_SRGB=43,
        VK_FORMAT_B8G8R8A8_SRGB=50,
        VK_FORMAT_R8G8B8A8_UNORM=37,
        VK_FORMAT_B8G8R8A8_UNORM=44,
    )
    with pytest.raises(ValueError, match="srgb, unorm, or auto"):
        _select_swapchain_format(vk, [43, 44], "linear")


def test_render_scale_is_bounded_by_runtime_limit() -> None:
    assert _scaled_dimension(1000, 1200, 0.5) == 500
    assert _scaled_dimension(1000, 1200, 2.0) == 1200
    assert _scaled_dimension(1, 1, 0.1) == 1


def test_presenter_validates_configuration() -> None:
    with pytest.raises(ValueError):
        OpenXrVulkanPresenter(OpenXrVulkanConfig(render_scale=0))


def test_openxr_defaults_to_validated_srgb_target_path() -> None:
    assert OpenXrVulkanConfig().swapchain_color_mode == "srgb"


def test_presenter_uses_controller_action_mixin_initializer() -> None:
    presenter = OpenXrVulkanPresenter()
    assert hasattr(presenter, "_init_controller_actions")
    assert not hasattr(presenter, "_initialize_controller_actions")


def test_openxr_frame_gate_waits_for_runtime_output_before_filament() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "if self._pending_output is None and not self._has_presented_frame:" in source
    assert "waiting for first runtime eye frame" in source
    assert "layer = self._render_projection_layer(views)" in source
    assert "bridge.set_screen_image(" not in source


def test_quad_layer_uses_runtime_output_size_and_openxr_visibility() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "_ensure_quad_swapchains(width, height)" in source
    assert "self.vulkan.copy_image(source, eye.resources[image_index])" in source
    assert "format_value if format_value is not None" in source
    assert "CompositionLayerQuad" in source
    assert "EyeVisibility.LEFT" in source
    assert "EyeVisibility.RIGHT" in source
    assert "_has_presented_frame" in source
    assert "self._last_quad_layers" in source
    assert "Render the world at the current headset pose" in source


def test_profile_pose_is_applied_once_to_openxr_reference_space() -> None:
    source = (Path(__file__).resolve().parents[1] /
              "src/xr_viewer/core_openxr_vulkan.py").read_text(encoding="utf-8")

    assert "_apply_profile_reference_space(views)" in source
    assert "self._profile_space_applied = True" in source
    assert "space_pose = raw_head @ np.linalg.inv(self._profile_head_transform)" in source
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
