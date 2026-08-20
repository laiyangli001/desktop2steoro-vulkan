"""Explicit GPU image interop boundary for capture and inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .vulkan_resources import VulkanImageResource


class VulkanInteropMode(str, Enum):
    NATIVE_EXTERNAL = "native_external"
    GPU_COPY = "gpu_copy"
    CPU_TEST_INPUT = "cpu_test_input"


@dataclass(frozen=True, slots=True)
class VulkanInteropCapabilities:
    """Capabilities reported by a producer-specific import adapter."""

    producer: str
    mode: VulkanInteropMode
    external_memory: bool = False
    external_semaphore: bool = False
    zero_copy: bool = False

    def __post_init__(self) -> None:
        if not str(self.producer).strip():
            raise ValueError("interop producer must not be empty")
        if self.mode is VulkanInteropMode.NATIVE_EXTERNAL and not self.external_memory:
            raise ValueError("native external interop requires external_memory")
        if self.zero_copy and not self.external_memory:
            raise ValueError("zero_copy interop requires external_memory")


@dataclass(frozen=True, slots=True)
class VulkanImageImportRequest:
    """Producer-owned image metadata handed to the Vulkan owner."""

    image: Any
    view: Any
    width: int
    height: int
    format: int
    layout: int
    access_mask: int
    stage_mask: int
    queue_family_index: int
    label: str
    external: bool = True


class VulkanImageImporter(Protocol):
    @property
    def capabilities(self) -> VulkanInteropCapabilities: ...

    def import_image(
        self, request: VulkanImageImportRequest
    ) -> VulkanImageResource: ...


class RegisteredImageImporter:
    """Register an already-created VkImage without taking native ownership."""

    def __init__(self, context: Any, capabilities: VulkanInteropCapabilities) -> None:
        self.context = context
        self._capabilities = capabilities

    @property
    def capabilities(self) -> VulkanInteropCapabilities:
        return self._capabilities

    def import_image(self, request: VulkanImageImportRequest) -> VulkanImageResource:
        resource = VulkanImageResource(
            context=self.context,
            image=request.image,
            view=request.view,
            width=request.width,
            height=request.height,
            format=request.format,
            layout=request.layout,
            access_mask=request.access_mask,
            stage_mask=request.stage_mask,
            queue_family_index=request.queue_family_index,
            external=request.external,
            label=request.label,
        )
        self.context.register_external_image(resource)
        return resource


class VulkanInteropSession:
    """Bounded lifetime manager for imported producer images."""

    def __init__(
        self,
        context: Any,
        importer: VulkanImageImporter,
        *,
        max_in_flight: int = 3,
    ) -> None:
        if int(max_in_flight) < 1:
            raise ValueError("max_in_flight must be at least one")
        self.context = context
        self.importer = importer
        self.max_in_flight = int(max_in_flight)
        self._resources: list[VulkanImageResource] = []
        self._closed = False

    @property
    def capabilities(self) -> VulkanInteropCapabilities:
        return self.importer.capabilities

    @property
    def in_flight_count(self) -> int:
        return len(self._resources)

    def import_frame(self, request: VulkanImageImportRequest) -> VulkanImageResource:
        if self._closed:
            raise RuntimeError("Vulkan interop session is closed")
        if len(self._resources) >= self.max_in_flight:
            raise RuntimeError("Vulkan interop frame budget is exhausted")
        resource = self.importer.import_image(request)
        if resource.context is not self.context:
            raise ValueError("importer returned an image from a different Vulkan context")
        self._resources.append(resource)
        return resource

    def release(self, resource: VulkanImageResource) -> None:
        if resource not in self._resources:
            raise ValueError("Vulkan interop resource is not in flight")
        self.context.unregister_external_image(resource)
        self._resources.remove(resource)

    def close(self) -> None:
        if self._closed:
            return
        for resource in tuple(self._resources):
            self.context.unregister_external_image(resource)
        self._resources.clear()
        self._closed = True
