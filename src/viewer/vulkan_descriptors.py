from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class VulkanDescriptorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DescriptorBudget:
    max_sets: int = 3
    storage_buffers_per_set: int = 1
    storage_images_per_set: int = 1

    def __post_init__(self) -> None:
        if self.max_sets < 1:
            raise ValueError("descriptor max_sets must be at least one")
        if self.storage_buffers_per_set < 0 or self.storage_images_per_set < 0:
            raise ValueError("descriptor counts cannot be negative")


@dataclass(frozen=True, slots=True)
class DescriptorBinding:
    binding: int
    descriptor_type: int
    descriptor_count: int = 1
    stage_flags: int = 0

    def __post_init__(self) -> None:
        if self.binding < 0 or self.descriptor_count < 1:
            raise ValueError("descriptor binding and count must be positive")


def create_descriptor_set_layout(context: Any, bindings: list[DescriptorBinding]) -> Any:
    vk = context.vk
    layout_bindings = [
        vk.VkDescriptorSetLayoutBinding(
            binding=item.binding,
            descriptorType=item.descriptor_type,
            descriptorCount=item.descriptor_count,
            stageFlags=item.stage_flags or vk.VK_SHADER_STAGE_COMPUTE_BIT,
            pImmutableSamplers=None,
        )
        for item in bindings
    ]
    return vk.vkCreateDescriptorSetLayout(
        context.device,
        vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(layout_bindings),
            pBindings=layout_bindings or None,
        ),
        None,
    )


class VulkanDescriptorArena:
    """Bounded descriptor pool used by graph passes."""

    def __init__(self, context: Any, budget: DescriptorBudget | None = None) -> None:
        self.context = context
        self.vk = context.vk
        self.budget = budget or DescriptorBudget()
        self.pool = None
        self._allocated = 0
        pool_sizes = []
        if self.budget.storage_buffers_per_set:
            pool_sizes.append(
                self.vk.VkDescriptorPoolSize(
                    type=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=self.budget.max_sets * self.budget.storage_buffers_per_set,
                )
            )
        if self.budget.storage_images_per_set:
            pool_sizes.append(
                self.vk.VkDescriptorPoolSize(
                    type=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=self.budget.max_sets * self.budget.storage_images_per_set,
                )
            )
        if not pool_sizes:
            raise ValueError("descriptor arena requires at least one descriptor type")
        self.pool = self.vk.vkCreateDescriptorPool(
            context.device,
            self.vk.VkDescriptorPoolCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags=self.vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                maxSets=self.budget.max_sets,
                poolSizeCount=len(pool_sizes),
                pPoolSizes=pool_sizes,
            ),
            None,
        )

    @property
    def allocated_sets(self) -> int:
        return self._allocated

    def allocate(self, layout: Any) -> Any:
        if self.pool is None:
            raise VulkanDescriptorError("descriptor arena is closed")
        if self._allocated >= self.budget.max_sets:
            raise VulkanDescriptorError("descriptor arena capacity exhausted")
        descriptor_set = self.vk.vkAllocateDescriptorSets(
            self.context.device,
            self.vk.VkDescriptorSetAllocateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=self.pool,
                descriptorSetCount=1,
                pSetLayouts=[layout],
            ),
        )[0]
        self._allocated += 1
        return descriptor_set

    def update_storage_buffer(self, descriptor_set: Any, binding: int, buffer: "VulkanStorageBuffer") -> None:
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=int(binding),
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[
                        self.vk.VkDescriptorBufferInfo(
                            buffer=buffer.buffer,
                            offset=0,
                            range=buffer.size,
                        )
                    ],
                )
            ],
            0,
            None,
        )

    def update_storage_image(self, descriptor_set: Any, binding: int, image: "VulkanStorageImage") -> None:
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=int(binding),
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=None,
                            imageView=image.view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_GENERAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )

    def close(self) -> None:
        if self.pool is not None and self.context.device is not None:
            self.vk.vkDestroyDescriptorPool(self.context.device, self.pool, None)
        self.pool = None
        self._allocated = 0

    def __enter__(self) -> "VulkanDescriptorArena":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanStorageBuffer:
    def __init__(self, context: Any, size: int) -> None:
        if int(size) < 1:
            raise ValueError("storage buffer size must be positive")
        self.context = context
        self.vk = context.vk
        self.size = int(size)
        self.buffer = None
        self.memory = None
        self._create()

    def _create(self) -> None:
        vk = self.vk
        self.buffer = vk.vkCreateBuffer(
            self.context.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=self.size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.context.device, self.buffer)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(self.context.physical_device)
        required = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        memory_type = next(
            (
                index
                for index, item in enumerate(properties.memoryTypes)
                if requirements.memoryTypeBits & (1 << index)
                and item.propertyFlags & required == required
            ),
            None,
        )
        if memory_type is None:
            raise VulkanDescriptorError("no host-visible coherent memory type for storage buffer")
        self.memory = vk.vkAllocateMemory(
            self.context.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=memory_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(self.context.device, self.buffer, self.memory, 0)

    def write_uint32(self, value: int) -> None:
        mapped = self.vk.vkMapMemory(self.context.device, self.memory, 0, 4, 0)
        mapped[:4] = int(value).to_bytes(4, "little")
        self.vk.vkUnmapMemory(self.context.device, self.memory)

    def read_uint32(self) -> int:
        mapped = self.vk.vkMapMemory(self.context.device, self.memory, 0, 4, 0)
        value = int.from_bytes(bytes(mapped[:4]), "little")
        self.vk.vkUnmapMemory(self.context.device, self.memory)
        return value

    def close(self) -> None:
        if self.context.device is not None:
            if self.buffer is not None:
                self.vk.vkDestroyBuffer(self.context.device, self.buffer, None)
            if self.memory is not None:
                self.vk.vkFreeMemory(self.context.device, self.memory, None)
        self.buffer = None
        self.memory = None

    def __enter__(self) -> "VulkanStorageBuffer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanStorageImage:
    def __init__(self, context: Any, width: int = 1, height: int = 1) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("storage image dimensions must be positive")
        self.context = context
        self.vk = context.vk
        self.width = int(width)
        self.height = int(height)
        self.image = None
        self.memory = None
        self.view = None
        self._create()

    def _create(self) -> None:
        vk = self.vk
        sharing_families = list(
            dict.fromkeys(
                (self.context.queue_family_index, self.context.compute_queue_family_index)
            )
        )
        sharing_mode = vk.VK_SHARING_MODE_CONCURRENT if len(sharing_families) > 1 else vk.VK_SHARING_MODE_EXCLUSIVE
        self.image = vk.vkCreateImage(
            self.context.device,
            vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
                mipLevels=1,
                arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                sharingMode=sharing_mode,
                queueFamilyIndexCount=len(sharing_families) if len(sharing_families) > 1 else 0,
                pQueueFamilyIndices=sharing_families if len(sharing_families) > 1 else None,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.context.device, self.image)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(self.context.physical_device)
        preferred = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        memory_type = next(
            (
                index
                for index, item in enumerate(properties.memoryTypes)
                if requirements.memoryTypeBits & (1 << index)
                and item.propertyFlags & preferred == preferred
            ),
            None,
        )
        if memory_type is None:
            raise VulkanDescriptorError("no device-local memory type for storage image")
        self.memory = vk.vkAllocateMemory(
            self.context.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=memory_type,
            ),
            None,
        )
        vk.vkBindImageMemory(self.context.device, self.image, self.memory, 0)
        self.view = vk.vkCreateImageView(
            self.context.device,
            vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=self.image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            ),
            None,
        )

    def transition_to_general(self) -> int:
        vk = self.vk

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=self.image,
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

        timeline_value = self.context.submit_on("compute", record)
        from viewer.vulkan_context import ImageState

        self.context.register_image_state(
            self.image,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                access_mask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                queue_family_index=self.context.compute_queue_family_index,
            ),
        )
        return timeline_value

    def close(self) -> None:
        if self.context.device is not None:
            if self.view is not None:
                self.vk.vkDestroyImageView(self.context.device, self.view, None)
            if self.image is not None:
                self.vk.vkDestroyImage(self.context.device, self.image, None)
            if self.memory is not None:
                self.vk.vkFreeMemory(self.context.device, self.memory, None)
        self.view = None
        self.image = None
        self.memory = None

    def __enter__(self) -> "VulkanStorageImage":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
