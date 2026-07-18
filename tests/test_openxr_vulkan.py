from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import xr

from viewer.vulkan_context import (
    VulkanCapabilityError,
    VulkanContext,
    VulkanUnavailableError,
    format_vulkan_version,
    make_vulkan_version,
    unpack_vulkan_version,
)
from xr_viewer.core_openxr_vulkan import (
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


def test_openxr_version_range_clamps_requested_vulkan_version() -> None:
    requirements = SimpleNamespace(
        min_api_version_supported=xr.Version(1, 1, 0),
        max_api_version_supported=xr.Version(1, 3, 0),
    )
    assert _select_vulkan_api_version(
        requirements, make_vulkan_version(1, 0, 0)
    ) == make_vulkan_version(1, 1, 0)
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


def test_render_scale_is_bounded_by_runtime_limit() -> None:
    assert _scaled_dimension(1000, 1200, 0.5) == 500
    assert _scaled_dimension(1000, 1200, 2.0) == 1200
    assert _scaled_dimension(1, 1, 0.1) == 1


def test_presenter_validates_configuration() -> None:
    with pytest.raises(ValueError):
        OpenXrVulkanPresenter(OpenXrVulkanConfig(render_scale=0))


def test_filament_bridge_binds_each_openxr_eye(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeBridge:
        def __init__(self, path):
            calls.append(("load", path))

        def create(self, **kwargs):
            calls.append(("create", kwargs["device"]))

        def create_swapchain(self, images, **kwargs):
            calls.append(("swapchain", (list(images), kwargs["format"])))

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

    assert len(presenter.filament_bridges) == 2
    assert calls == [
        ("load", "bridge.dll"),
        ("create", 3),
        ("swapchain", (["left-image"], 43)),
        ("load", "bridge.dll"),
        ("create", 3),
        ("swapchain", (["right-image"], 43)),
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


def test_swapchain_image_is_not_released_when_wait_fails() -> None:
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
    assert calls == ["acquire", "wait"]


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


def test_standalone_vulkan_context_smoke() -> None:
    try:
        context = VulkanContext.create()
    except (VulkanUnavailableError, VulkanCapabilityError) as exc:
        pytest.skip(str(exc))
    try:
        assert context.device_info.name
        assert context.device_info.queue_family_index >= 0
    finally:
        context.close()
    assert context.closed
