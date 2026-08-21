"""Vulkan Compute wrapper for the GPU RGB/RGBA to NV12 intermediate pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .vulkan_compute_pipeline import VulkanComputePipeline
from .vulkan_descriptors import DescriptorBinding
from .vulkan_resources import VulkanTransientImage


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


class VulkanRgbToNv12Intermediate:
    """Reusable GPU images produced before the NV12 Video image copy."""

    @staticmethod
    def dimensions(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        VulkanRgbToNv12Pipeline.dispatch_size(width, height)
        return (int(width), int(height)), (int(width) // 2, int(height) // 2)

    def __init__(self, context: Any, width: int, height: int) -> None:
        self.context = context
        self.vk = context.vk
        (y_width, y_height), (uv_width, uv_height) = self.dimensions(width, height)
        usage = (
            self.vk.VK_IMAGE_USAGE_STORAGE_BIT
            | self.vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | self.vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        )
        self.y = VulkanTransientImage(
            context, y_width, y_height,
            format=self.vk.VK_FORMAT_R8_UNORM,
            label="stream-nv12-y-intermediate",
            usage=usage,
        )
        try:
            self.uv = VulkanTransientImage(
                context, uv_width, uv_height,
                format=self.vk.VK_FORMAT_R8G8_UNORM,
                label="stream-nv12-uv-intermediate",
                usage=usage,
            )
        except Exception:
            self.y.close()
            raise

    def close(self) -> None:
        self.uv.close()
        self.y.close()
