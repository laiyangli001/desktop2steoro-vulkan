from __future__ import annotations

import os
import ctypes
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VulkanImageResource:
    """Non-owning contract for an imported or externally managed Vulkan image."""

    context: Any
    image: Any
    view: Any | None
    width: int
    height: int
    format: int
    layout: int
    access_mask: int
    stage_mask: int
    queue_family_index: int
    mip_levels: int = 1
    external: bool = True
    label: str = "external-image"

    def __post_init__(self) -> None:
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("Vulkan image dimensions must be positive")
        if self.image is None:
            raise ValueError("Vulkan image handle is required")
        if int(self.mip_levels) < 1:
            raise ValueError("Vulkan image mip level count must be positive")
        if not str(self.label).strip():
            raise ValueError("Vulkan image label must not be empty")

    def require_view(self) -> Any:
        if self.view is None:
            raise ValueError(f"Vulkan image view is required for {self.label}")
        return self.view


class VulkanTransientImage:
    """Own a device-local image used only inside the Vulkan presentation path."""

    def __init__(
        self,
        context: Any,
        width: int,
        height: int,
        *,
        format: int,
        label: str,
        mip_levels: int = 1,
        usage: int | None = None,
        aspect_mask: int | None = None,
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("transient image dimensions must be positive")
        if not str(label).strip():
            raise ValueError("transient image label must not be empty")
        max_mip_levels = int(math.floor(math.log2(max(int(width), int(height))))) + 1
        if int(mip_levels) < 1 or int(mip_levels) > max_mip_levels:
            raise ValueError("transient image mip level count is invalid")
        self.context = context
        self.vk = context.vk
        self.width = int(width)
        self.height = int(height)
        self.format = int(format)
        self.label = str(label)
        self.mip_levels = int(mip_levels)
        self._usage = usage
        self._aspect_mask = aspect_mask
        self.image = None
        self.memory = None
        self.view = None
        self.resource: VulkanImageResource | None = None
        self._create()

    def _create(self) -> None:
        vk = self.vk
        sharing_families = list(
            dict.fromkeys((self.context.queue_family_index, self.context.compute_queue_family_index))
        )
        sharing_mode = (
            vk.VK_SHARING_MODE_CONCURRENT
            if len(sharing_families) > 1
            else vk.VK_SHARING_MODE_EXCLUSIVE
        )
        self.image = vk.vkCreateImage(
            self.context.device,
            vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=self.format,
                extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
                mipLevels=self.mip_levels,
                arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=self._usage if self._usage is not None else (
                    vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                    | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                    | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                ),
                sharingMode=sharing_mode,
                queueFamilyIndexCount=len(sharing_families) if len(sharing_families) > 1 else 0,
                pQueueFamilyIndices=sharing_families if len(sharing_families) > 1 else None,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.context.device, self.image)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(self.context.physical_device)
        memory_type = next(
            (
                index
                for index, item in enumerate(properties.memoryTypes)
                if requirements.memoryTypeBits & (1 << index)
                and item.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            ),
            None,
        )
        if memory_type is None:
            raise RuntimeError("no device-local memory type for transient image")
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
                format=self.format,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=(
                        self._aspect_mask
                        if self._aspect_mask is not None
                        else vk.VK_IMAGE_ASPECT_COLOR_BIT
                    ),
                    baseMipLevel=0,
                    levelCount=self.mip_levels,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
            ),
            None,
        )
        self.resource = VulkanImageResource(
            context=self.context,
            image=self.image,
            view=self.view,
            width=self.width,
            height=self.height,
            format=self.format,
            layout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            access_mask=0,
            stage_mask=0,
            queue_family_index=self.context.queue_family_index,
            mip_levels=self.mip_levels,
            external=False,
            label=self.label,
        )
        self.context.register_external_image(self.resource)

    def close(self) -> None:
        if self.resource is not None:
            try:
                self.context.unregister_external_image(self.resource)
            except Exception:
                pass
        if self.context.device is not None:
            if self.view is not None:
                self.vk.vkDestroyImageView(self.context.device, self.view, None)
            if self.image is not None:
                self.vk.vkDestroyImage(self.context.device, self.image, None)
            if self.memory is not None:
                self.vk.vkFreeMemory(self.context.device, self.memory, None)
        self.resource = None
        self.view = None
        self.image = None
        self.memory = None


class VulkanDepthAttachment(VulkanTransientImage):
    """Owned single-sample depth image that can be borrowed by Filament."""

    def __init__(self, context: Any, width: int, height: int, *, label: str):
        vk = context.vk
        selected = None
        for candidate in (
            getattr(vk, "VK_FORMAT_D24_UNORM_S8_UINT", None),
            getattr(vk, "VK_FORMAT_D32_SFLOAT", None),
            getattr(vk, "VK_FORMAT_D16_UNORM", None),
        ):
            if candidate is None:
                continue
            properties = vk.vkGetPhysicalDeviceFormatProperties(
                context.physical_device, candidate
            )
            if int(properties.optimalTilingFeatures) & int(
                vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
            ):
                selected = int(candidate)
                break
        if selected is None:
            raise RuntimeError("no Vulkan depth attachment format is supported")
        aspect = vk.VK_IMAGE_ASPECT_DEPTH_BIT
        if selected == getattr(vk, "VK_FORMAT_D24_UNORM_S8_UINT", -1):
            aspect |= vk.VK_IMAGE_ASPECT_STENCIL_BIT
        super().__init__(
            context,
            width,
            height,
            format=selected,
            label=label,
            mip_levels=1,
            usage=(
                vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                | vk.VK_IMAGE_USAGE_SAMPLED_BIT
            ),
            aspect_mask=aspect,
        )


class VulkanExternalImageRegistry:
    """Registers external image state without taking ownership of image resources."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._resources: dict[int, VulkanImageResource] = {}

    def register(self, resource: VulkanImageResource) -> None:
        if resource.context is not self.context:
            raise ValueError("external image belongs to a different Vulkan context")
        key = id(resource.image)
        if key in self._resources:
            raise ValueError(f"external image is already registered: {resource.label}")
        from viewer.vulkan_context import ImageState

        self.context.register_image_state(
            resource.image,
            ImageState(
                layout=resource.layout,
                access_mask=resource.access_mask,
                stage_mask=resource.stage_mask,
                queue_family_index=resource.queue_family_index,
            ),
        )
        self._resources[key] = resource

    def unregister(self, resource: VulkanImageResource) -> None:
        key = id(resource.image)
        current = self._resources.get(key)
        if current is not resource:
            raise ValueError("external image is not registered by this registry")
        self.context.unregister_image_state(resource.image)
        del self._resources[key]

    def close(self) -> None:
        for resource in tuple(self._resources.values()):
            self.unregister(resource)

    def discard(self) -> None:
        """Forget non-owning handles after context teardown cannot unregister state."""

        self._resources.clear()

    @property
    def registered_count(self) -> int:
        return len(self._resources)


class VulkanHostImage:
    """Host-visible linear image used to upload small Vulkan overlay textures."""

    def __init__(self, context: Any, width: int, height: int, *, format: int, label: str):
        self.context = context
        self.vk = context.vk
        self.width = int(width)
        self.height = int(height)
        self.format = int(format)
        self.label = str(label)
        vk = self.vk
        self.image = vk.vkCreateImage(
            context.device,
            vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=self.format,
                extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
                mipLevels=1,
                arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_LINEAR,
                usage=vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_PREINITIALIZED,
            ),
            None,
        )
        requirements = vk.vkGetImageMemoryRequirements(context.device, self.image)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(context.physical_device)
        flags = int(vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        memory_type = next(
            (index for index, item in enumerate(properties.memoryTypes)
             if requirements.memoryTypeBits & (1 << index)
             and int(item.propertyFlags) & flags == flags),
            None,
        )
        if memory_type is None:
            vk.vkDestroyImage(context.device, self.image, None)
            raise RuntimeError("no host-visible Vulkan memory type for overlay upload")
        self.memory = vk.vkAllocateMemory(
            context.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=memory_type,
            ),
            None,
        )
        vk.vkBindImageMemory(context.device, self.image, self.memory, 0)
        subresource = vk.VkImageSubresource(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0, arrayLayer=0
        )
        self._layout = vk.vkGetImageSubresourceLayout(context.device, self.image, subresource)
        self.resource = VulkanImageResource(
            context=context,
            image=self.image,
            view=None,
            width=self.width,
            height=self.height,
            format=self.format,
            layout=vk.VK_IMAGE_LAYOUT_PREINITIALIZED,
            access_mask=vk.VK_ACCESS_HOST_WRITE_BIT,
            stage_mask=vk.VK_PIPELINE_STAGE_HOST_BIT,
            queue_family_index=context.queue_family_index,
            external=True,
            label=self.label,
        )
        context.register_external_image(self.resource)

    def upload(self, rgba) -> None:
        import numpy as np

        pixels = np.ascontiguousarray(rgba, dtype=np.uint8)
        if pixels.shape != (self.height, self.width, 4):
            raise ValueError("overlay upload dimensions must match VulkanHostImage")
        vk = self.vk
        mapped = vk.vkMapMemory(self.context.device, self.memory, 0, self._layout.size, 0)
        try:
            row_pitch = int(self._layout.rowPitch)
            row_bytes = self.width * 4
            for row in range(self.height):
                start = int(self._layout.offset) + row * row_pitch
                # PyVulkan exposes mapped host memory as a writable buffer.
                # Keep the write on that buffer; casting the returned cffi
                # object as an integer pointer fails on current PyVulkan.
                mapped[start:start + row_bytes] = pixels[row].tobytes()
        finally:
            vk.vkUnmapMemory(self.context.device, self.memory)

    def read_rgba(self):
        """Read a completed linear image for explicit visual diagnostics."""
        import numpy as np

        vk = self.vk
        mapped = vk.vkMapMemory(self.context.device, self.memory, 0, self._layout.size, 0)
        try:
            row_pitch = int(self._layout.rowPitch)
            row_bytes = self.width * 4
            pixels = np.empty((self.height, self.width, 4), dtype=np.uint8)
            for row in range(self.height):
                start = int(self._layout.offset) + row * row_pitch
                pixels[row] = np.frombuffer(
                    bytes(mapped[start:start + row_bytes]), dtype=np.uint8
                ).reshape(self.width, 4)
            return pixels
        finally:
            vk.vkUnmapMemory(self.context.device, self.memory)

    def close(self) -> None:
        try:
            self.context.unregister_external_image(self.resource)
        except Exception:
            pass
        self.vk.vkDestroyImage(self.context.device, self.image, None)
        self.vk.vkFreeMemory(self.context.device, self.memory, None)


class VulkanExportableBuffer:
    """Own a Vulkan storage buffer whose memory can be imported by a GPU producer."""

    def __init__(self, context: Any, size: int, *, label: str) -> None:
        if int(size) < 1:
            raise ValueError("exportable buffer size must be positive")
        if not str(label).strip():
            raise ValueError("exportable buffer label must not be empty")
        self.context = context
        self.vk = context.vk
        self.size = int(size)
        self.label = str(label)
        self.buffer = None
        self.memory = None
        self.allocation_size = 0
        self._export_handle = None
        self._handle_type = self._resolve_handle_type()
        self._create()

    def _resolve_handle_type(self) -> int:
        if os.name == "nt":
            return int(self.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
        if os.name == "posix":
            return int(self.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
        raise RuntimeError(f"unsupported external-memory platform: {os.name}")

    def _create(self) -> None:
        vk = self.vk
        sharing_families = list(
            dict.fromkeys(
                (
                    self.context.queue_family_index,
                    self.context.compute_queue_family_index,
                )
            )
        )
        sharing_mode = (
            vk.VK_SHARING_MODE_CONCURRENT
            if len(sharing_families) > 1
            else vk.VK_SHARING_MODE_EXCLUSIVE
        )
        external_buffer = vk.VkExternalMemoryBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            handleTypes=self._handle_type,
        )
        self.buffer = vk.vkCreateBuffer(
            self.context.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext=external_buffer,
                size=self.size,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=sharing_mode,
                queueFamilyIndexCount=len(sharing_families) if len(sharing_families) > 1 else 0,
                pQueueFamilyIndices=sharing_families if len(sharing_families) > 1 else None,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.context.device, self.buffer)
        self.allocation_size = int(requirements.size)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(self.context.physical_device)
        memory_type = next(
            (
                index
                for index, item in enumerate(properties.memoryTypes)
                if requirements.memoryTypeBits & (1 << index)
                and item.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            ),
            None,
        )
        if memory_type is None:
            vk.vkDestroyBuffer(self.context.device, self.buffer, None)
            self.buffer = None
            raise RuntimeError("no device-local memory type for exportable buffer")
        export_memory = vk.VkExportMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            handleTypes=self._handle_type,
        )
        self.memory = vk.vkAllocateMemory(
            self.context.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext=export_memory,
                allocationSize=requirements.size,
                memoryTypeIndex=memory_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(self.context.device, self.buffer, self.memory, 0)

    @property
    def handle_type(self) -> int:
        return self._handle_type

    @property
    def export_handle(self) -> Any:
        if self._export_handle is not None:
            return self._export_handle
        vk = self.vk
        if os.name == "nt":
            proc = vk.lib.vkGetDeviceProcAddr(
                self.context.device, b"vkGetMemoryWin32HandleKHR"
            )
            if proc == vk.ffi.NULL:
                raise RuntimeError("vkGetMemoryWin32HandleKHR is unavailable")
            function = vk.ffi.cast(
                "VkResult(*)(VkDevice, const VkMemoryGetWin32HandleInfoKHR*, void**)",
                proc,
            )
            output = vk.ffi.new("void **")
            info = vk.ffi.new("VkMemoryGetWin32HandleInfoKHR *")
            info.sType = vk.VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR
            info.memory = self.memory
            info.handleType = self._handle_type
            result = function(self.context.device, info, output)
            if int(result) != int(vk.VK_SUCCESS):
                raise RuntimeError(f"vkGetMemoryWin32HandleKHR failed: {result}")
            self._export_handle = int(vk.ffi.cast("uintptr_t", output[0]))
            return self._export_handle

        proc = vk.lib.vkGetDeviceProcAddr(self.context.device, b"vkGetMemoryFdKHR")
        if proc == vk.ffi.NULL:
            raise RuntimeError("vkGetMemoryFdKHR is unavailable")
        function = vk.ffi.cast(
            "VkResult(*)(VkDevice, const VkMemoryGetFdInfoKHR*, int*)", proc
        )
        output = vk.ffi.new("int *")
        info = vk.ffi.new("VkMemoryGetFdInfoKHR *")
        info.sType = vk.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR
        info.memory = self.memory
        info.handleType = self._handle_type
        result = function(self.context.device, info, output)
        if int(result) != int(vk.VK_SUCCESS):
            raise RuntimeError(f"vkGetMemoryFdKHR failed: {result}")
        self._export_handle = int(output[0])
        return self._export_handle

    def close_export_handle(self) -> None:
        if self._export_handle is None:
            return
        if os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(int(self._export_handle)))
        else:
            os.close(int(self._export_handle))
        self._export_handle = None

    def close(self) -> None:
        self.close_export_handle()
        if self.context.device is not None:
            if self.buffer is not None:
                self.vk.vkDestroyBuffer(self.context.device, self.buffer, None)
            if self.memory is not None:
                self.vk.vkFreeMemory(self.context.device, self.memory, None)
        self.buffer = None
        self.memory = None
        self.allocation_size = 0

    def __enter__(self) -> "VulkanExportableBuffer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanExportableImage:
    """Own one Vulkan image whose memory can be imported by a GPU producer."""

    def __init__(
        self,
        context: Any,
        width: int,
        height: int,
        *,
        label: str,
        format: int | None = None,
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("exportable image dimensions must be positive")
        if not str(label).strip():
            raise ValueError("exportable image label must not be empty")
        self.context = context
        self.vk = context.vk
        self.width = int(width)
        self.height = int(height)
        self.label = str(label)
        self.format = int(format or self.vk.VK_FORMAT_R8G8B8A8_UNORM)
        self.image = None
        self.memory = None
        self.allocation_size = 0
        self.view = None
        self.resource: VulkanImageResource | None = None
        self._export_handle = None
        self._handle_type = self._resolve_handle_type()
        self._create()

    @staticmethod
    def required_device_extensions() -> tuple[str, ...]:
        if os.name == "nt":
            return (
                "VK_KHR_external_memory",
                "VK_KHR_external_memory_win32",
            )
        if os.name == "posix":
            return (
                "VK_KHR_external_memory",
                "VK_KHR_external_memory_fd",
            )
        raise RuntimeError(f"unsupported external-memory platform: {os.name}")

    @staticmethod
    def optional_external_semaphore_extensions() -> tuple[str, ...]:
        if os.name == "nt":
            return ("VK_KHR_external_semaphore", "VK_KHR_external_semaphore_win32")
        if os.name == "posix":
            return ("VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd")
        return ()

    def _resolve_handle_type(self) -> int:
        if os.name == "nt":
            return int(self.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
        return int(self.vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)

    def _create(self) -> None:
        vk = self.vk
        sharing_families = list(
            dict.fromkeys(
                (
                    self.context.queue_family_index,
                    self.context.compute_queue_family_index,
                )
            )
        )
        sharing_mode = (
            vk.VK_SHARING_MODE_CONCURRENT
            if len(sharing_families) > 1
            else vk.VK_SHARING_MODE_EXCLUSIVE
        )
        external_image = vk.VkExternalMemoryImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            handleTypes=self._handle_type,
        )
        self.image = vk.vkCreateImage(
            self.context.device,
            vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext=external_image,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=self.format,
                extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
                mipLevels=1,
                arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(
                    vk.VK_IMAGE_USAGE_STORAGE_BIT
                    | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                ),
                sharingMode=sharing_mode,
                queueFamilyIndexCount=len(sharing_families) if len(sharing_families) > 1 else 0,
                pQueueFamilyIndices=sharing_families if len(sharing_families) > 1 else None,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.context.device, self.image)
        self.allocation_size = int(requirements.size)
        properties = vk.vkGetPhysicalDeviceMemoryProperties(self.context.physical_device)
        memory_type = next(
            (
                index
                for index, item in enumerate(properties.memoryTypes)
                if requirements.memoryTypeBits & (1 << index)
                and item.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            ),
            None,
        )
        if memory_type is None:
            raise RuntimeError("no device-local memory type for exportable image")
        export_memory = vk.VkExportMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            handleTypes=self._handle_type,
        )
        self.memory = vk.vkAllocateMemory(
            self.context.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext=export_memory,
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
                format=self.format,
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
        self.resource = VulkanImageResource(
            context=self.context,
            image=self.image,
            view=self.view,
            width=self.width,
            height=self.height,
            format=self.format,
            layout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            access_mask=0,
            stage_mask=0,
            queue_family_index=self.context.queue_family_index,
            external=True,
            label=self.label,
        )
        self.context.register_external_image(self.resource)

    @property
    def handle_type(self) -> int:
        return self._handle_type

    def require_view(self) -> Any:
        if self.view is None:
            raise RuntimeError("exportable image view is unavailable")
        return self.view

    @property
    def export_handle(self) -> Any:
        if self._export_handle is None:
            self._export_handle = self._get_export_handle()
        return self._export_handle

    def _get_export_handle(self) -> Any:
        vk = self.vk
        if os.name == "nt":
            # PyVulkan exposes the raw loader symbol but rejects extension names
            # that are not in its generated dispatch table. Use the raw symbol so
            # platform external-memory entry points remain available.
            proc = vk.lib.vkGetDeviceProcAddr(
                self.context.device, b"vkGetMemoryWin32HandleKHR"
            )
            if proc == vk.ffi.NULL:
                raise RuntimeError("vkGetMemoryWin32HandleKHR is unavailable")
            function = vk.ffi.cast(
                "VkResult(*)(VkDevice, const VkMemoryGetWin32HandleInfoKHR*, void**)" ,
                proc,
            )
            output = vk.ffi.new("void **")
            info = vk.ffi.new("VkMemoryGetWin32HandleInfoKHR *")
            info.sType = vk.VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR
            info.memory = self.memory
            info.handleType = self._handle_type
            result = function(
                self.context.device,
                info,
                output,
            )
            if int(result) != int(vk.VK_SUCCESS):
                raise RuntimeError(f"vkGetMemoryWin32HandleKHR failed: {result}")
            return int(vk.ffi.cast("uintptr_t", output[0]))

        proc = vk.lib.vkGetDeviceProcAddr(self.context.device, b"vkGetMemoryFdKHR")
        if proc == vk.ffi.NULL:
            raise RuntimeError("vkGetMemoryFdKHR is unavailable")
        function = vk.ffi.cast(
            "VkResult(*)(VkDevice, const VkMemoryGetFdInfoKHR*, int*)", proc
        )
        output = vk.ffi.new("int *")
        info = vk.ffi.new("VkMemoryGetFdInfoKHR *")
        info.sType = vk.VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR
        info.memory = self.memory
        info.handleType = self._handle_type
        result = function(
            self.context.device,
            info,
            output,
        )
        if int(result) != int(vk.VK_SUCCESS):
            raise RuntimeError(f"vkGetMemoryFdKHR failed: {result}")
        return int(output[0])

    def close_export_handle(self) -> None:
        if self._export_handle is None:
            return
        if os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(int(self._export_handle)))
        else:
            os.close(int(self._export_handle))
        self._export_handle = None

    def close(self) -> None:
        if self.resource is not None:
            try:
                self.context.unregister_external_image(self.resource)
            except Exception:
                pass
        self.close_export_handle()
        if self.context.device is not None:
            if self.view is not None:
                self.vk.vkDestroyImageView(self.context.device, self.view, None)
            if self.image is not None:
                self.vk.vkDestroyImage(self.context.device, self.image, None)
            if self.memory is not None:
                self.vk.vkFreeMemory(self.context.device, self.memory, None)
        self.resource = None
        self.view = None
        self.image = None
        self.memory = None
        self.allocation_size = 0

    def __enter__(self) -> "VulkanExportableImage":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanExportableSemaphore:
    """Own a Vulkan semaphore exported to an external GPU producer."""

    @staticmethod
    def required_device_extensions() -> tuple[str, ...]:
        if os.name == "nt":
            return (
                "VK_KHR_external_semaphore",
                "VK_KHR_external_semaphore_win32",
            )
        if os.name == "posix":
            return (
                "VK_KHR_external_semaphore",
                "VK_KHR_external_semaphore_fd",
            )
        raise RuntimeError(f"unsupported external-semaphore platform: {os.name}")

    def __init__(
        self,
        context: Any,
        *,
        label: str = "external-semaphore",
        timeline: bool = False,
    ) -> None:
        self.context = context
        self.vk = context.vk
        self.label = str(label)
        self.timeline = bool(timeline)
        self._handle_type = self._resolve_handle_type()
        self.semaphore = None
        self._export_handle = None
        export_info = self.vk.VkExportSemaphoreCreateInfo(
            sType=self.vk.VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            handleTypes=self._handle_type,
        )
        create_next = export_info
        if self.timeline:
            create_next = self.vk.VkSemaphoreTypeCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                pNext=export_info,
                semaphoreType=self.vk.VK_SEMAPHORE_TYPE_TIMELINE,
                initialValue=0,
            )
        self.semaphore = self.vk.vkCreateSemaphore(
            context.device,
            self.vk.VkSemaphoreCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext=create_next,
            ),
            None,
        )

    def _resolve_handle_type(self) -> int:
        if os.name == "nt":
            return int(self.vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT)
        if os.name == "posix":
            return int(self.vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT)
        raise RuntimeError(f"unsupported external-semaphore platform: {os.name}")

    @property
    def export_handle(self) -> Any:
        if self._export_handle is None:
            vk = self.vk
            if os.name == "nt":
                proc = vk.lib.vkGetDeviceProcAddr(
                    self.context.device, b"vkGetSemaphoreWin32HandleKHR"
                )
                if proc == vk.ffi.NULL:
                    raise RuntimeError("vkGetSemaphoreWin32HandleKHR is unavailable")
                function = vk.ffi.cast(
                    "VkResult(*)(VkDevice, const VkSemaphoreGetWin32HandleInfoKHR*, void**)" ,
                    proc,
                )
                output = vk.ffi.new("void **")
                info = vk.ffi.new("VkSemaphoreGetWin32HandleInfoKHR *")
                info.sType = vk.VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR
                info.semaphore = self.semaphore
                info.handleType = self._handle_type
                result = function(self.context.device, info, output)
                if int(result) != int(vk.VK_SUCCESS):
                    raise RuntimeError(f"vkGetSemaphoreWin32HandleKHR failed: {result}")
                self._export_handle = int(vk.ffi.cast("uintptr_t", output[0]))
            else:
                proc = vk.lib.vkGetDeviceProcAddr(
                    self.context.device, b"vkGetSemaphoreFdKHR"
                )
                if proc == vk.ffi.NULL:
                    raise RuntimeError("vkGetSemaphoreFdKHR is unavailable")
                function = vk.ffi.cast(
                    "VkResult(*)(VkDevice, const VkSemaphoreGetFdInfoKHR*, int*)", proc
                )
                output = vk.ffi.new("int *")
                info = vk.ffi.new("VkSemaphoreGetFdInfoKHR *")
                info.sType = vk.VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR
                info.semaphore = self.semaphore
                info.handleType = self._handle_type
                result = function(self.context.device, info, output)
                if int(result) != int(vk.VK_SUCCESS):
                    raise RuntimeError(f"vkGetSemaphoreFdKHR failed: {result}")
                self._export_handle = int(output[0])
        return self._export_handle

    def close_export_handle(self) -> None:
        if self._export_handle is None:
            return
        if os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(int(self._export_handle)))
        else:
            os.close(int(self._export_handle))
        self._export_handle = None

    def close(self) -> None:
        self.close_export_handle()
        if self.semaphore is not None and self.context.device is not None:
            self.vk.vkDestroySemaphore(self.context.device, self.semaphore, None)
        self.semaphore = None

    def __enter__(self) -> "VulkanExportableSemaphore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class VulkanBinarySemaphore:
    """Own a device-local binary semaphore used between Vulkan submissions."""

    def __init__(self, context: Any, *, label: str = "binary-semaphore") -> None:
        self.context = context
        self.vk = context.vk
        self.label = str(label)
        self.semaphore = self.vk.vkCreateSemaphore(
            context.device,
            self.vk.VkSemaphoreCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            ),
            None,
        )

    def close(self) -> None:
        if self.semaphore is not None and self.context.device is not None:
            self.vk.vkDestroySemaphore(self.context.device, self.semaphore, None)
        self.semaphore = None

    def __enter__(self) -> "VulkanBinarySemaphore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
