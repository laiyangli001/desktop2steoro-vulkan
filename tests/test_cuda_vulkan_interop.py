import ctypes
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from app_runtime.gpu_producer import (
    GpuProducerAdapter,
    GpuProducerUnavailableError,
    _ADAPTER_FACTORIES,
    create_gpu_producer_adapter,
)
from app_runtime.runtime_output import (
    CudaVulkanOutputAdapter,
    RocmVulkanOutputAdapter,
)
from viewer.cuda_vulkan_interop import (
    _ExternalSemaphoreWaitParams,
    _ExternalSemaphoreSignalParams,
    _SemaphoreSignalParams,
)
from viewer.vulkan_context import VulkanContext, VulkanContextConfig
from viewer.vulkan_resources import VulkanExportableImage
from viewer.vulkan_resources import VulkanImageResource


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
    assert _ExternalSemaphoreWaitParams.params.offset == 0
    assert _ExternalSemaphoreWaitParams.flags.offset == 72
    assert ctypes.sizeof(_ExternalSemaphoreWaitParams) == 144


def test_cuda_external_semaphore_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", raising=False)
    assert not CudaVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "1")
    assert CudaVulkanOutputAdapter._external_semaphore_requested()


def test_rocm_external_semaphore_is_capability_gated_by_default(monkeypatch) -> None:
    monkeypatch.delenv("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", raising=False)
    assert RocmVulkanOutputAdapter._external_semaphore_requested()
    monkeypatch.setenv("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", "0")
    assert not RocmVulkanOutputAdapter._external_semaphore_requested()


def test_cuda_output_adapter_implements_backend_neutral_gpu_contract() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    assert isinstance(adapter, GpuProducerAdapter)
    assert adapter.backend_name == "cuda"
    assert adapter.output_sync_mode == "gpu_synchronized"
    assert adapter.external_semaphore_sync_mode == "gpu_external_semaphore"


def test_gpu_producer_factory_selects_cuda_without_importing_vendor_api() -> None:
    adapter = create_gpu_producer_adapter(SimpleNamespace(), backend="cuda")
    assert isinstance(adapter, CudaVulkanOutputAdapter)


def test_gpu_producer_factory_rejects_unregistered_backend() -> None:
    assert _ADAPTER_FACTORIES["rocm"] is RocmVulkanOutputAdapter


def test_gpu_producer_factory_reports_unknown_backend() -> None:
    with pytest.raises(GpuProducerUnavailableError, match="unknown"):
        create_gpu_producer_adapter(SimpleNamespace(), backend="unknown")


def test_vulkan_output_slot_waits_for_consumer_release() -> None:
    adapter = CudaVulkanOutputAdapter.__new__(CudaVulkanOutputAdapter)
    adapter._lease_condition = threading.Condition()
    adapter._active_leases = {}
    adapter._closed = False
    adapter._claim_slot(0, 10)
    claimed = threading.Event()

    def claim_reused_slot() -> None:
        adapter._claim_slot(0, 11)
        claimed.set()

    worker = threading.Thread(target=claim_reused_slot)
    worker.start()
    assert not claimed.wait(0.05)
    adapter.release_frame(10)
    worker.join(timeout=1.0)
    assert claimed.is_set()
    adapter.release_frame(11)


def test_output_contract_publishes_actual_source_layout_and_queue_family() -> None:
    vk = SimpleNamespace(
        VK_IMAGE_LAYOUT_GENERAL=1,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL=2,
    )
    state = SimpleNamespace(layout=1, queue_family_index=4)
    context = SimpleNamespace(
        vk=vk,
        image_state=lambda _image: state,
    )
    resource = VulkanImageResource(
        context=context,
        image=object(),
        view=None,
        width=2,
        height=2,
        format=37,
        layout=1,
        access_mask=0,
        stage_mask=0,
        queue_family_index=4,
    )

    contract = GpuProducerAdapter.source_image_contract(resource)
    assert contract == {"layout": "general", "queue_family": 4}

    state.layout = 2
    assert CudaVulkanOutputAdapter._source_image_contract(resource)["layout"] == (
        "shader_read_only_optimal"
    )


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
