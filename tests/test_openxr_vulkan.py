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
