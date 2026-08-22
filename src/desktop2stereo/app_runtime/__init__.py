"""Application lifecycle and runtime assembly."""

from .bootstrap import main
from .gpu_producer import (
    GpuProducerAdapter,
    GpuProducerUnavailableError,
    create_gpu_producer_adapter,
)
from .output_contract import LatestFrameOutputRouter, VulkanStereoOutputFrame
from .runtime_entry import run_processing_runtime
from .vulkan_runtime import VulkanDeviceLostError, VulkanRuntimeConfig, VulkanRuntimeSession

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
