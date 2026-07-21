import ctypes
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from app_runtime.runtime_output import CudaVulkanOutputAdapter
from viewer.cuda_vulkan_interop import (
    _ExternalSemaphoreSignalParams,
    _SemaphoreSignalParams,
)
from viewer.vulkan_context import VulkanContext, VulkanContextConfig
from viewer.vulkan_resources import VulkanExportableImage


def test_cuda_external_semaphore_signal_params_match_runtime_abi() -> None:
    # CUDA's v2 signal parameter layout is 72 bytes of nested params,
    # followed by flags and sixteen reserved uint32 values.
    assert _SemaphoreSignalParams.fence_value.offset == 0
    assert _SemaphoreSignalParams.nv_sci_sync.offset == 8
    assert _SemaphoreSignalParams.keyed_mutex_key.offset == 16
    assert _SemaphoreSignalParams.reserved.offset == 24
    assert ctypes.sizeof(_SemaphoreSignalParams) == 72
    assert _ExternalSemaphoreSignalParams.params.offset == 0
    assert _ExternalSemaphoreSignalParams.flags.offset == 72
    assert ctypes.sizeof(_ExternalSemaphoreSignalParams) == 144


def test_cuda_external_semaphore_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", raising=False)
    assert not CudaVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "1")
    assert CudaVulkanOutputAdapter._external_semaphore_requested()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_cuda_tensor_reaches_vulkan_output_slot_without_cpu_roundtrip() -> None:
    context = VulkanContext.create(
        VulkanContextConfig(
            required_device_extensions=VulkanExportableImage.required_device_extensions()
        )
    )
    destination = None
    adapter = None
    try:
        presenter = SimpleNamespace(initialized=True, vulkan=context)
        adapter = CudaVulkanOutputAdapter(presenter)
        result = SimpleNamespace(
            left_eye=torch.full((32, 48, 4), 7, dtype=torch.uint8, device="cuda"),
            right_eye=torch.full((32, 48, 4), 11, dtype=torch.uint8, device="cuda"),
            debug_info={"output_format": "openxr_full_synthesis_eyes"},
        )
        frame = adapter.convert(result, frame_id=3, timestamp=1.0)

        assert frame.left_eye.width == 48
        assert frame.left_eye.height == 32
        assert frame.left_eye.format == context.vk.VK_FORMAT_R8G8B8A8_SRGB
        assert len(adapter.left_slots) == 3
        assert len(adapter.right_slots) == 3
        assert frame.metadata["vulkan_output_ring_slot"] == 0
        assert frame.metadata["vulkan_output_ring_size"] == 3
        assert context.image_state(frame.left_eye.image).layout == context.vk.VK_IMAGE_LAYOUT_GENERAL

        destination = VulkanExportableImage(
            context,
            48,
            32,
            label="cuda-output-destination",
            format=context.vk.VK_FORMAT_R8G8B8A8_SRGB,
        )
        timeline = context.copy_image(frame.left_eye, destination.resource)
        context.wait_idle()
        assert timeline > 0
    finally:
        if adapter is not None:
            adapter.close()
        if destination is not None:
            destination.close()
        context.close()
