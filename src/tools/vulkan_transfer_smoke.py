"""Validate a Vulkan-only image copy into an output-compatible layout."""

from __future__ import annotations

from viewer.vulkan_context import ImageState, VulkanContext
from viewer.vulkan_descriptors import VulkanStorageImage
from viewer.vulkan_resources import VulkanImageResource


def _transition_to_graphics_general(context: VulkanContext, image: VulkanStorageImage) -> None:
    vk = context.vk

    def record(command_buffer):
        barrier = vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            command_buffer,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            [barrier],
        )

    context.submit_on("graphics", record)
    context.register_image_state(
        image.image,
        ImageState(
            layout=vk.VK_IMAGE_LAYOUT_GENERAL,
            access_mask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            stage_mask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            queue_family_index=context.queue_family_index,
        ),
    )


def main() -> int:
    context = VulkanContext.create()
    source = VulkanStorageImage(context, width=16, height=8)
    destination = VulkanStorageImage(context, width=16, height=8)
    try:
        _transition_to_graphics_general(context, source)
        _transition_to_graphics_general(context, destination)
        source_resource = VulkanImageResource(
            context=context,
            image=source.image,
            view=source.view,
            width=source.width,
            height=source.height,
            format=context.vk.VK_FORMAT_R8G8B8A8_UNORM,
            layout=context.vk.VK_IMAGE_LAYOUT_GENERAL,
            access_mask=context.vk.VK_ACCESS_SHADER_WRITE_BIT,
            stage_mask=context.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            queue_family_index=context.queue_family_index,
            label="transfer-source",
        )
        destination_resource = VulkanImageResource(
            context=context,
            image=destination.image,
            view=destination.view,
            width=destination.width,
            height=destination.height,
            format=context.vk.VK_FORMAT_R8G8B8A8_UNORM,
            layout=context.vk.VK_IMAGE_LAYOUT_GENERAL,
            access_mask=context.vk.VK_ACCESS_SHADER_WRITE_BIT,
            stage_mask=context.vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            queue_family_index=context.queue_family_index,
            label="transfer-destination",
        )
        timeline = context.copy_image(source_resource, destination_resource)
        context.wait_idle()
        state = context.image_state(destination.image)
        if state.layout != context.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            raise RuntimeError("destination image did not reach color attachment layout")
        print(f"vulkan_transfer_smoke: PASS timeline={timeline} state=ready")
        return 0
    finally:
        destination.close()
        source.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
