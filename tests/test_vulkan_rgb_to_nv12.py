from __future__ import annotations

import pytest

from viewer.vulkan_rgb_to_nv12 import VulkanRgbToNv12Intermediate, VulkanRgbToNv12Pipeline


def test_rgb_to_nv12_dispatch_uses_8x8_workgroups() -> None:
    assert VulkanRgbToNv12Pipeline.dispatch_size(3840, 2160) == (480, 270, 1)


def test_rgb_to_nv12_intermediate_dimensions_match_nv12() -> None:
    assert VulkanRgbToNv12Intermediate.dimensions(3840, 2160) == ((3840, 2160), (1920, 1080))


@pytest.mark.parametrize("width,height", [(1, 2), (3, 2), (2, 3), (0, 0)])
def test_rgb_to_nv12_requires_even_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="even"):
        VulkanRgbToNv12Pipeline.dispatch_size(width, height)
