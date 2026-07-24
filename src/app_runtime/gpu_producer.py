"""Backend-neutral contract for GPU producers feeding the Vulkan presenter."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

from viewer.vulkan_resources import VulkanImageResource


class GpuProducerAdapter(ABC):
    """Adapt a vendor GPU result into the presenter Vulkan image contract.

    Vendor APIs stay inside concrete adapters.  The presenter only consumes
    Vulkan resources, synchronization metadata, and a release callback.
    """

    backend_name = "unknown"

    @property
    def output_sync_mode(self) -> str:
        """Return the backend-neutral synchronization mode for one output."""
        return "gpu_synchronized"

    @property
    def external_semaphore_sync_mode(self) -> str:
        """Return the backend-neutral mode for producer-ready semaphores."""
        return "gpu_external_semaphore"

    @staticmethod
    def source_image_contract(resource: VulkanImageResource) -> dict[str, object]:
        """Publish the actual Vulkan layout and queue family of a source image."""
        context = resource.context
        state = context.image_state(resource.image)
        vk = context.vk
        layout = int(state.layout)
        if layout == int(vk.VK_IMAGE_LAYOUT_GENERAL):
            layout_name = "general"
        elif layout == int(vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL):
            layout_name = "shader_read_only_optimal"
        else:
            layout_name = f"vulkan_{layout}"
        return {
            "layout": layout_name,
            "queue_family": int(state.queue_family_index),
        }

    @abstractmethod
    def convert(self, runtime_result: Any, *, frame_id: int, timestamp: float):
        """Convert a backend result without a CPU pixel round trip."""

    def release_frame(self, frame_id: int) -> None:
        """Release a frame after the presenter has stopped sampling it."""

    def close(self) -> None:
        """Release backend resources owned by the adapter."""


class GpuProducerUnavailableError(RuntimeError):
    """Raised when no safe adapter exists for the selected GPU backend."""


_ADAPTER_FACTORIES: dict[str, type[GpuProducerAdapter]] = {}


def register_gpu_producer_adapter(
    backend_name: str, adapter_type: type[GpuProducerAdapter]
) -> None:
    """Register one concrete producer without exposing vendor APIs to callers."""
    key = str(backend_name).strip().lower()
    if not key:
        raise ValueError("GPU producer backend name must not be empty")
    if not issubclass(adapter_type, GpuProducerAdapter):
        raise TypeError("GPU producer adapter must inherit GpuProducerAdapter")
    _ADAPTER_FACTORIES[key] = adapter_type


def create_gpu_producer_adapter(presenter, *, backend: str | None = None) -> GpuProducerAdapter:
    """Create the selected producer adapter through the backend-neutral registry.

    The built-in registration is lazy so importing the contract does not load
    CUDA, HIP, or any other vendor runtime.  A future ROCm/HIP module can
    register itself under ``rocm`` without changing the Presenter.
    """
    if not _ADAPTER_FACTORIES:
        # Importing this module registers the currently available adapters.
        from . import runtime_output  # noqa: F401

    selected = str(
        backend or os.environ.get("D2S_VULKAN_GPU_BACKEND", "auto")
    ).strip().lower()
    if selected in {"", "auto", "default"}:
        try:
            import torch

            selected = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        except Exception:
            selected = "cuda"
    adapter_type = _ADAPTER_FACTORIES.get(selected)
    if adapter_type is None:
        available = ", ".join(sorted(_ADAPTER_FACTORIES)) or "none"
        raise GpuProducerUnavailableError(
            f"no Vulkan GPU producer adapter for '{selected}' (available: {available})"
        )
    return adapter_type(presenter)
