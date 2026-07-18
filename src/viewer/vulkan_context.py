from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Iterable


class VulkanUnavailableError(RuntimeError):
    pass


class VulkanCapabilityError(RuntimeError):
    pass


def make_vulkan_version(major: int, minor: int, patch: int = 0) -> int:
    return (int(major) << 22) | (int(minor) << 12) | int(patch)


def unpack_vulkan_version(version: int) -> tuple[int, int, int]:
    value = int(version)
    return value >> 22, (value >> 12) & 0x3FF, value & 0xFFF


def format_vulkan_version(version: int) -> str:
    return ".".join(str(part) for part in unpack_vulkan_version(version))


@dataclass(frozen=True, slots=True)
class VulkanDeviceInfo:
    name: str
    api_version: int
    driver_version: int
    vendor_id: int
    device_id: int
    device_type: int
    queue_family_index: int

    @property
    def api_version_text(self) -> str:
        return format_vulkan_version(self.api_version)


@dataclass(frozen=True, slots=True)
class VulkanContextConfig:
    application_name: str = "Desktop2Stereo Vulkan"
    engine_name: str = "D2S"
    api_version: int = make_vulkan_version(1, 2, 0)
    enable_validation: bool = False
    required_instance_extensions: tuple[str, ...] = ()
    required_device_extensions: tuple[str, ...] = ()


class VulkanContext:
    def __init__(
        self,
        *,
        vk: Any,
        instance: Any,
        physical_device: Any,
        device: Any,
        queue: Any,
        queue_family_index: int,
        device_info: VulkanDeviceInfo,
        owns_instance: bool,
        owns_device: bool,
    ) -> None:
        self.vk = vk
        self.instance = instance
        self.physical_device = physical_device
        self.device = device
        self.queue = queue
        self.queue_family_index = int(queue_family_index)
        self.device_info = device_info
        self._owns_instance = bool(owns_instance)
        self._owns_device = bool(owns_device)
        self._command_pool = None
        self._command_buffer = None
        self._fence = None
        self._known_image_layouts: dict[int, int] = {}
        self._lock = RLock()
        self._closed = False
        self._create_command_resources()

    @classmethod
    def create(cls, config: VulkanContextConfig | None = None) -> "VulkanContext":
        cfg = config or VulkanContextConfig()
        vk = _import_vulkan()
        available_layers = _enumerate_names(vk.vkEnumerateInstanceLayerProperties(), "layerName")
        available_extensions = _enumerate_names(
            vk.vkEnumerateInstanceExtensionProperties(None),
            "extensionName",
        )
        required_instance_extensions = _require_names(
            "Vulkan instance extension",
            cfg.required_instance_extensions,
            available_extensions,
        )
        layers: tuple[str, ...] = ()
        if cfg.enable_validation:
            validation_layer = "VK_LAYER_KHRONOS_validation"
            if validation_layer not in available_layers:
                raise VulkanCapabilityError(
                    f"Vulkan validation requested but {validation_layer} is unavailable"
                )
            layers = (validation_layer,)

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=cfg.application_name,
            applicationVersion=1,
            pEngineName=cfg.engine_name,
            engineVersion=1,
            apiVersion=int(cfg.api_version),
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=list(layers) or None,
            enabledExtensionCount=len(required_instance_extensions),
            ppEnabledExtensionNames=list(required_instance_extensions) or None,
        )

        instance = None
        device = None
        try:
            instance = vk.vkCreateInstance(create_info, None)
            physical_device, queue_family_index = _select_physical_device(vk, instance)
            available_device_extensions = _enumerate_names(
                vk.vkEnumerateDeviceExtensionProperties(physical_device, None),
                "extensionName",
            )
            required_device_extensions = _require_names(
                "Vulkan device extension",
                cfg.required_device_extensions,
                available_device_extensions,
            )
            device = _create_device(
                vk,
                physical_device,
                queue_family_index,
                required_device_extensions,
            )
            queue = vk.vkGetDeviceQueue(device, queue_family_index, 0)
            info = _device_info(vk, physical_device, queue_family_index)
            return cls(
                vk=vk,
                instance=instance,
                physical_device=physical_device,
                device=device,
                queue=queue,
                queue_family_index=queue_family_index,
                device_info=info,
                owns_instance=True,
                owns_device=True,
            )
        except Exception:
            if device is not None:
                vk.vkDestroyDevice(device, None)
            if instance is not None:
                vk.vkDestroyInstance(instance, None)
            raise

    @classmethod
    def adopt(
        cls,
        *,
        instance: Any,
        physical_device: Any,
        device: Any,
        queue_family_index: int,
        owns_instance: bool = True,
        owns_device: bool = True,
    ) -> "VulkanContext":
        vk = _import_vulkan()
        queue = vk.vkGetDeviceQueue(device, int(queue_family_index), 0)
        return cls(
            vk=vk,
            instance=instance,
            physical_device=physical_device,
            device=device,
            queue=queue,
            queue_family_index=int(queue_family_index),
            device_info=_device_info(vk, physical_device, int(queue_family_index)),
            owns_instance=owns_instance,
            owns_device=owns_device,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def command_pool(self) -> Any:
        self._ensure_open()
        return self._command_pool

    def image_handle_from_address(self, address: int) -> Any:
        self._ensure_open()
        if not address:
            raise ValueError("VkImage address must be non-zero")
        return self.vk.ffi.cast("VkImage", int(address))

    def clear_color_image(
        self,
        image: Any,
        color: tuple[float, float, float, float],
    ) -> None:
        if len(color) != 4:
            raise ValueError("clear color must contain four components")
        vk = self.vk
        image_key = _cffi_handle_address(vk, image)
        old_layout = self._known_image_layouts.get(
            image_key,
            vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )

        def record(command_buffer: Any) -> None:
            source_access = (
                0
                if old_layout == vk.VK_IMAGE_LAYOUT_UNDEFINED
                else vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            )
            source_stage = (
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                if old_layout == vk.VK_IMAGE_LAYOUT_UNDEFINED
                else vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            )
            to_transfer = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=source_access,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=old_layout,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=_color_subresource_range(vk),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                source_stage,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [to_transfer],
            )
            vk.vkCmdClearColorImage(
                command_buffer,
                image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                vk.VkClearColorValue(float32=list(float(component) for component in color)),
                1,
                [_color_subresource_range(vk)],
            )
            to_runtime = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=_color_subresource_range(vk),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [to_runtime],
            )

        self.submit(record)
        self._known_image_layouts[image_key] = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL

    def submit(self, record: Callable[[Any], None]) -> None:
        with self._lock:
            self._ensure_open()
            vk = self.vk
            vk.vkResetFences(self.device, 1, [self._fence])
            vk.vkResetCommandBuffer(self._command_buffer, 0)
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            )
            vk.vkBeginCommandBuffer(self._command_buffer, begin_info)
            try:
                record(self._command_buffer)
                vk.vkEndCommandBuffer(self._command_buffer)
            except Exception:
                try:
                    vk.vkEndCommandBuffer(self._command_buffer)
                except Exception:
                    pass
                raise
            submit_info = vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[self._command_buffer],
            )
            vk.vkQueueSubmit(self.queue, 1, [submit_info], self._fence)
            vk.vkWaitForFences(self.device, 1, [self._fence], vk.VK_TRUE, 10_000_000_000)

    def wait_idle(self) -> None:
        with self._lock:
            if not self._closed and self.device is not None:
                self.vk.vkDeviceWaitIdle(self.device)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            vk = self.vk
            try:
                if self.device is not None:
                    vk.vkDeviceWaitIdle(self.device)
            finally:
                if self.device is not None:
                    if self._fence is not None:
                        vk.vkDestroyFence(self.device, self._fence, None)
                    if self._command_pool is not None:
                        vk.vkDestroyCommandPool(self.device, self._command_pool, None)
                    if self._owns_device:
                        vk.vkDestroyDevice(self.device, None)
                if self.instance is not None and self._owns_instance:
                    vk.vkDestroyInstance(self.instance, None)
                self._known_image_layouts.clear()
                self._command_buffer = None
                self._command_pool = None
                self._fence = None
                self.queue = None
                self.device = None
                self.physical_device = None
                self.instance = None
                self._closed = True

    def __enter__(self) -> "VulkanContext":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _create_command_resources(self) -> None:
        vk = self.vk
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex=self.queue_family_index,
        )
        self._command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
        allocation_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self._command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        self._command_buffer = vk.vkAllocateCommandBuffers(
            self.device,
            allocation_info,
        )[0]
        fence_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        )
        self._fence = vk.vkCreateFence(self.device, fence_info, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VulkanContext is closed")


def _import_vulkan() -> Any:
    try:
        import vulkan as vk
    except (ImportError, OSError) as exc:
        raise VulkanUnavailableError(
            "Python Vulkan bindings or the Vulkan loader are unavailable"
        ) from exc
    return vk


def _enumerate_names(properties: Iterable[Any], field: str) -> set[str]:
    names: set[str] = set()
    for item in properties:
        value = getattr(item, field)
        if isinstance(value, bytes):
            value = value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        names.add(str(value))
    return names


def _require_names(
    label: str,
    required: Iterable[str],
    available: set[str],
) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(name) for name in required))
    missing = [name for name in requested if name not in available]
    if missing:
        raise VulkanCapabilityError(f"Missing {label}s: {', '.join(missing)}")
    return requested


def _select_physical_device(vk: Any, instance: Any) -> tuple[Any, int]:
    devices = vk.vkEnumeratePhysicalDevices(instance)
    if not devices:
        raise VulkanCapabilityError("No Vulkan physical device is available")
    candidates: list[tuple[int, Any, int]] = []
    for physical_device in devices:
        queue_family_index = _find_graphics_queue_family(vk, physical_device)
        if queue_family_index is None:
            continue
        properties = vk.vkGetPhysicalDeviceProperties(physical_device)
        score = {
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 300,
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 200,
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: 100,
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU: 10,
        }.get(int(properties.deviceType), 0)
        candidates.append((score, physical_device, queue_family_index))
    if not candidates:
        raise VulkanCapabilityError("No Vulkan device exposes a graphics queue")
    _, physical_device, queue_family_index = max(candidates, key=lambda item: item[0])
    return physical_device, queue_family_index


def find_graphics_queue_family(vk: Any, physical_device: Any) -> int:
    queue_family_index = _find_graphics_queue_family(vk, physical_device)
    if queue_family_index is None:
        raise VulkanCapabilityError("OpenXR Vulkan device has no graphics queue")
    return queue_family_index


def _find_graphics_queue_family(vk: Any, physical_device: Any) -> int | None:
    for index, properties in enumerate(
        vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    ):
        if properties.queueCount and properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
            return index
    return None


def _device_info(
    vk: Any,
    physical_device: Any,
    queue_family_index: int,
) -> VulkanDeviceInfo:
    properties = vk.vkGetPhysicalDeviceProperties(physical_device)
    return VulkanDeviceInfo(
        name=str(properties.deviceName),
        api_version=int(properties.apiVersion),
        driver_version=int(properties.driverVersion),
        vendor_id=int(properties.vendorID),
        device_id=int(properties.deviceID),
        device_type=int(properties.deviceType),
        queue_family_index=int(queue_family_index),
    )


def _create_device(
    vk: Any,
    physical_device: Any,
    queue_family_index: int,
    required_extensions: Iterable[str],
) -> Any:
    extension_names = tuple(required_extensions)
    queue_info = vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=int(queue_family_index),
        queueCount=1,
        pQueuePriorities=[1.0],
    )
    device_info = vk.VkDeviceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=[queue_info],
        enabledExtensionCount=len(extension_names),
        ppEnabledExtensionNames=list(extension_names) or None,
    )
    return vk.vkCreateDevice(physical_device, device_info, None)


def _color_subresource_range(vk: Any) -> Any:
    return vk.VkImageSubresourceRange(
        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
        baseMipLevel=0,
        levelCount=1,
        baseArrayLayer=0,
        layerCount=1,
    )


def _cffi_handle_address(vk: Any, handle: Any) -> int:
    return int(vk.ffi.cast("uintptr_t", handle))
