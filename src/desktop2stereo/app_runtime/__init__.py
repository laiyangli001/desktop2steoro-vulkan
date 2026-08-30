"""Application lifecycle and runtime assembly."""

from .bootstrap import main
from .gpu_producer import (
    GpuProducerAdapter,
    GpuProducerUnavailableError,
    create_gpu_producer_adapter,
)
from .output_contract import LatestFrameOutputRouter, VulkanStereoOutputFrame
from .vulkan_runtime import VulkanDeviceLostError, VulkanRuntimeConfig, VulkanRuntimeSession


def run_processing_runtime(*args, **kwargs):
    """Load the processing runtime only when the runtime is actually started.

    GUI startup must not resolve runtime display settings before the user has
    had a chance to refresh or select an available output display.
    """
    from .runtime_entry import run_processing_runtime as _run_processing_runtime

    return _run_processing_runtime(*args, **kwargs)

__all__ = [
    "LatestFrameOutputRouter",
    "GpuProducerAdapter",
    "GpuProducerUnavailableError",
    "VulkanDeviceLostError",
    "VulkanRuntimeConfig",
    "VulkanRuntimeSession",
    "VulkanStereoOutputFrame",
    "main",
    "create_gpu_producer_adapter",
    "run_processing_runtime",
]
