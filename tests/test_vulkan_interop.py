from types import SimpleNamespace

import pytest

from viewer.vulkan_interop import (
    RegisteredImageImporter,
    VulkanImageImportRequest,
    VulkanInteropCapabilities,
    VulkanInteropMode,
    VulkanInteropSession,
)


class FakeContext:
    def __init__(self) -> None:
        self.registered = []

    def register_external_image(self, resource) -> None:
        self.registered.append(resource)

    def unregister_external_image(self, resource) -> None:
        self.registered.remove(resource)


def _request(label: str) -> VulkanImageImportRequest:
    return VulkanImageImportRequest(
        image=SimpleNamespace(name=label),
        view=SimpleNamespace(name=f"{label}-view"),
        width=8,
        height=4,
        format=37,
        layout=general_layout,
        access_mask=0,
        stage_mask=0,
        queue_family_index=0,
        label=label,
    )


general_layout = 1


def test_registered_importer_preserves_non_owning_resource_boundary() -> None:
    context = FakeContext()
    importer = RegisteredImageImporter(
        context,
        VulkanInteropCapabilities(
            producer="openxr",
            mode=VulkanInteropMode.NATIVE_EXTERNAL,
            external_memory=True,
            zero_copy=True,
        ),
    )
    session = VulkanInteropSession(context, importer, max_in_flight=1)
    resource = session.import_frame(_request("eye-0"))

    assert session.capabilities.zero_copy is True
    assert resource.external is True
    assert session.in_flight_count == 1
    with pytest.raises(RuntimeError, match="frame budget"):
        session.import_frame(_request("eye-1"))

    session.release(resource)
    assert context.registered == []


def test_interop_close_is_idempotent() -> None:
    context = FakeContext()
    importer = RegisteredImageImporter(
        context,
        VulkanInteropCapabilities(
            producer="cuda",
            mode=VulkanInteropMode.GPU_COPY,
        ),
    )
    session = VulkanInteropSession(context, importer)
    session.import_frame(_request("frame"))
    session.close()
    session.close()
    assert session.in_flight_count == 0
    assert context.registered == []


def test_sampling_transition_is_explicitly_separate_from_cuda_prepare() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/viewer/vulkan_context.py").read_text(
        encoding="utf-8"
    )
    assert "def prepare_external_image_for_sampling" in source
    assert "VK_IMAGE_LAYOUT_GENERAL" in source
    assert "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL" in source
    assert "def release_external_image_from_sampling" in source
    assert "wait_semaphore: Any | None = None" in source
    assert "signal_semaphore: Any | None = None" in source
    assert "pWaitSemaphores=wait_semaphores or None" in source
    assert "pSignalSemaphores=signal_semaphores or None" in source
