"""Vulkan Compute wrapper for the GPU RGB/RGBA to NV12 intermediate pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .vulkan_compute_pipeline import VulkanComputePipeline
from .vulkan_descriptors import DescriptorBinding


class VulkanRgbToNv12Pipeline:
    """Record the RGB -> Y/UV conversion without CPU staging.

    The Y and UV images are intermediate R8/RG8 resources. The native bridge
    must copy them into its profile-compatible NV12 multi-plane encode image;
    this class intentionally does not pretend that two storage images are a
    valid Vulkan Video source.
    """

    WORKGROUP_X = 8
    WORKGROUP_Y = 8

    @staticmethod
    def shader_path() -> Path:
        return Path(__file__).resolve().parents[1] / "shaders" / "d2s_rgb_to_nv12.spv"

    @staticmethod
    def descriptor_bindings(context: Any) -> list[DescriptorBinding]:
        vk = context.vk
        return [
            DescriptorBinding(0, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
            DescriptorBinding(1, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
            DescriptorBinding(2, vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
        ]

    def __init__(self, context: Any) -> None:
        self.context = context
        self.pipeline = VulkanComputePipeline(
            context,
            self.shader_path(),
            descriptor_bindings=self.descriptor_bindings(context),
        )

    @staticmethod
    def dispatch_size(width: int, height: int) -> tuple[int, int, int]:
        if int(width) < 2 or int(height) < 2 or int(width) % 2 or int(height) % 2:
            raise ValueError("RGB to NV12 conversion requires positive even dimensions")
        return (
            (int(width) + VulkanRgbToNv12Pipeline.WORKGROUP_X - 1) // VulkanRgbToNv12Pipeline.WORKGROUP_X,
            (int(height) + VulkanRgbToNv12Pipeline.WORKGROUP_Y - 1) // VulkanRgbToNv12Pipeline.WORKGROUP_Y,
            1,
        )

    def record(self, command_buffer: Any, *, width: int, height: int, descriptor_set: Any) -> None:
        self.pipeline.record_dispatch(
            command_buffer,
            group_count_x=self.dispatch_size(width, height)[0],
            group_count_y=self.dispatch_size(width, height)[1],
            descriptor_set=descriptor_set,
        )

    def close(self) -> None:
        self.pipeline.close()
