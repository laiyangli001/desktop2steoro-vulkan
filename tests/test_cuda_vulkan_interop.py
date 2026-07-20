from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from app_runtime.runtime_output import CudaVulkanOutputAdapter
from viewer.vulkan_context import VulkanContext, VulkanContextConfig
from viewer.vulkan_resources import VulkanExportableImage


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
        assert context.image_state(frame.left_eye.image).layout == context.vk.VK_IMAGE_LAYOUT_GENERAL

        destination = VulkanExportableImage(context, 48, 32, label="cuda-output-destination")
        timeline = context.copy_image(frame.left_eye, destination.resource)
        context.wait_idle()
        assert timeline > 0
    finally:
        if adapter is not None:
            adapter.close()
        if destination is not None:
            destination.close()
        context.close()
