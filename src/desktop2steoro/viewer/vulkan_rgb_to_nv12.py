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
        self._y_layout = self.vk.VK_IMAGE_LAYOUT_UNDEFINED
        self._uv_layout = self.vk.VK_IMAGE_LAYOUT_UNDEFINED

    def close(self) -> None:
        self.uv.close()
        self.y.close()

    def record_frame(
        self,
        command_buffer: Any,
        *,
        pipeline: VulkanRgbToNv12Pipeline,
        width: int,
        height: int,
        descriptor_set: Any,
        destination_image: Any,
    ) -> None:
        """Record one complete GPU conversion and Video-image submission prep."""
        self.record_prepare_for_compute(command_buffer)
        pipeline.record(
            command_buffer,
            width=width,
            height=height,
            descriptor_set=descriptor_set,
        )
        self.record_copy_to_video_nv12(command_buffer, destination_image)

    def record_copy_to_video_nv12(self, command_buffer: Any, destination_image: Any) -> None:
        """Record GPU-only Y/UV copies into one multi-plane NV12 image.

        ``destination_image`` is the image from FFmpeg's profile-compatible
        `AV_PIX_FMT_VULKAN` frame. The caller owns its synchronization timeline.
        No host-visible mapping or CPU pixel copy is performed here.
        """
        vk = self.vk
        barriers = [
            vk.VkImageMemoryBarrier2(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                srcStageMask=vk.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                srcAccessMask=vk.VK_ACCESS_2_SHADER_WRITE_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                dstAccessMask=vk.VK_ACCESS_2_TRANSFER_READ_BIT,
                oldLayout=self._y_layout if resource is self.y.resource else self._uv_layout,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            for resource in (self.y.resource, self.uv.resource)
        ]
        barriers.extend(
            vk.VkImageMemoryBarrier2(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                srcStageMask=vk.VK_PIPELINE_STAGE_2_NONE,
                srcAccessMask=vk.VK_ACCESS_2_NONE,
                dstStageMask=vk.VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                dstAccessMask=vk.VK_ACCESS_2_TRANSFER_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=destination_image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=aspect,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            for aspect in (vk.VK_IMAGE_ASPECT_PLANE_0_BIT, vk.VK_IMAGE_ASPECT_PLANE_1_BIT)
        )
        vk.vkCmdPipelineBarrier2(
            command_buffer,
            vk.VkDependencyInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                imageMemoryBarrierCount=len(barriers),
                pImageMemoryBarriers=barriers,
            ),
        )
        copies = [
            vk.VkImageCopy(
                srcSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0, baseArrayLayer=0, layerCount=1,
                ),
                dstSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=aspect,
                    mipLevel=0, baseArrayLayer=0, layerCount=1,
                ),
                extent=vk.VkExtent3D(width=width, height=height, depth=1),
            )
            for resource, aspect, width, height in (
                (self.y, vk.VK_IMAGE_ASPECT_PLANE_0_BIT, self.y.width, self.y.height),
                (self.uv, vk.VK_IMAGE_ASPECT_PLANE_1_BIT, self.uv.width, self.uv.height),
            )
        ]
        for resource, copy in zip((self.y, self.uv), copies):
            vk.vkCmdCopyImage(
                command_buffer,
                resource.image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                destination_image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [copy],
            )
        final_barrier = vk.VkImageMemoryBarrier2(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            srcStageMask=vk.VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            srcAccessMask=vk.VK_ACCESS_2_TRANSFER_WRITE_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
            dstAccessMask=vk.VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=destination_image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_PLANE_0_BIT | vk.VK_IMAGE_ASPECT_PLANE_1_BIT,
                baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier2(
            command_buffer,
            vk.VkDependencyInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                imageMemoryBarrierCount=1,
                pImageMemoryBarriers=[final_barrier],
            ),
        )
        self._y_layout = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        self._uv_layout = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL

    def record_prepare_for_compute(self, command_buffer: Any) -> None:
        """Transition reusable Y/UV images back to compute-write layout."""
        vk = self.vk
        barriers = []
        for resource, old_layout in ((self.y, self._y_layout), (self.uv, self._uv_layout)):
            barriers.append(vk.VkImageMemoryBarrier2(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                srcStageMask=vk.VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                srcAccessMask=vk.VK_ACCESS_2_TRANSFER_READ_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                dstAccessMask=vk.VK_ACCESS_2_SHADER_WRITE_BIT,
                oldLayout=old_layout,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            ))
        vk.vkCmdPipelineBarrier2(
            command_buffer,
            vk.VkDependencyInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                imageMemoryBarrierCount=len(barriers),
                pImageMemoryBarriers=barriers,
            ),
        )
        self._y_layout = vk.VK_IMAGE_LAYOUT_GENERAL
        self._uv_layout = vk.VK_IMAGE_LAYOUT_GENERAL
