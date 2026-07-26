from __future__ import annotations

import torch

from app_runtime.runtime_output import VulkanHostOutputAdapter


def test_vulkan_host_output_adapter_detects_hwc_and_chw_extents():
    assert VulkanHostOutputAdapter._tensor_extent(torch.zeros(2160, 3840, 4)) == (
        3840,
        2160,
    )
    assert VulkanHostOutputAdapter._tensor_extent(torch.zeros(3, 2160, 3840)) == (
        3840,
        2160,
    )


def test_vulkan_host_output_adapter_converts_float_chw_to_rgba8():
    tensor = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[1.0, 0.5], [0.0, 0.25]],
            [[0.25, 0.5], [0.75, 0.25]],
        ],
        dtype=torch.float32,
    )

    rgba = VulkanHostOutputAdapter._tensor_to_rgba(tensor, width=2, height=2)

    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype.name == "uint8"
    assert rgba[0, 0].tolist() == [0, 255, 64, 255]
    assert rgba[1, 1].tolist() == [64, 64, 64, 255]


def test_vulkan_host_output_adapter_preserves_uint8_bytes():
    tensor = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ],
        dtype=torch.uint8,
    )
    rgba = VulkanHostOutputAdapter._tensor_to_rgba(tensor, width=2, height=2)

    assert rgba[0, 0].tolist() == [1, 5, 9, 255]
    assert rgba[1, 1].tolist() == [4, 8, 12, 255]
