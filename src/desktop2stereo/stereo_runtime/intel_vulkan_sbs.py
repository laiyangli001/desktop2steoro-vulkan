"""GPU-only bridge from Vulkan eye images to Intel D3D11 final-SBS input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stereo_runtime.providers.intel.d3d11_sbs_surface import D3D11SbsSurface


@dataclass(frozen=True, slots=True)
class IntelVulkanSbsFrame:
    """Borrowed final BGRA texture contract consumed by the native encoder."""

    texture: int
    device: int
    width: int
    height: int
    adapter_luid: int
    gpu_to_cpu: bool = False
    zero_copy: bool = False
    gpu_copy_count: int = 2


class IntelVulkanSbsComposer:
    """Compose Vulkan left/right images into a D3D11-owned shared texture.

    The current bridge uses two Vulkan blits because the stereo shader produces
    separate eye images. There is no CPU readback, but the two GPU blits mean
    this is explicitly a GPU-only copy path, not strict zero-copy yet.
    """

    def __init__(self, context: Any, eye_width: int, eye_height: int) -> None:
        self.context = context
        self.eye_width = int(eye_width)
        self.eye_height = int(eye_height)
        if self.eye_width < 1 or self.eye_height < 1:
            raise ValueError("Intel Vulkan SBS eye dimensions must be positive")
        adapter_luid = int(getattr(context.device_info, "adapter_luid", 0))
        if adapter_luid == 0:
            raise RuntimeError("Vulkan Adapter LUID is unavailable for Intel SBS composition")
        self.adapter_luid = adapter_luid
        self.surface = D3D11SbsSurface(
            width=self.eye_width * 2,
            height=self.eye_height,
            adapter_luid=adapter_luid,
        )
        if self.surface.adapter_luid != adapter_luid:
            self.surface.close()
            raise RuntimeError("D3D11 SBS surface Adapter LUID does not match Vulkan")
        self.imported = None
        try:
            self.imported = self.surface.import_bgra_into_vulkan(context)
        except Exception:
            self.surface.close()
            raise

    def compose(
        self,
        left_resource: Any,
        right_resource: Any,
        *,
        ready_timeline: int | None = None,
    ) -> IntelVulkanSbsFrame:
        if self.imported is None or self.imported.resource is None:
            raise RuntimeError("Intel Vulkan SBS imported destination is unavailable")
        if getattr(left_resource, "context", None) is not self.context or getattr(
            right_resource, "context", None
        ) is not self.context:
            raise RuntimeError("Vulkan SBS eye resources belong to a different context")
        timeline = self.context.compose_sbs_images(
            left_resource,
            right_resource,
            self.imported.resource,
            wait_for_timeline=ready_timeline,
        )
        # D3D11 has no Vulkan semaphore object in this bridge yet. Waiting for
        # the submitted Vulkan timeline is still GPU synchronization; it does
        # not map or copy image data through the CPU. The encoder consumer will
        # issue the D3D11 VideoProcessor conversion exactly once.
        self.context.wait_for_timeline(timeline)
        return IntelVulkanSbsFrame(
            texture=self.surface.bgra_texture,
            device=self.surface.device,
            width=self.surface.width,
            height=self.surface.height,
            adapter_luid=self.adapter_luid,
        )

    def close(self) -> None:
        if self.imported is not None:
            self.imported.close()
            self.imported = None
        if self.surface is not None:
            self.surface.close()
            self.surface = None


__all__ = ["IntelVulkanSbsComposer", "IntelVulkanSbsFrame"]
