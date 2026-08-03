from types import SimpleNamespace

import pytest

from viewer.vulkan_context import VulkanContext

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


def test_external_image_release_short_circuits_after_device_loss() -> None:
    context = object.__new__(VulkanContext)
    context._device_lost = True

    assert context.release_external_image_from_sampling(object()) == 0


def test_submit_on_latches_device_loss_from_any_vulkan_stage() -> None:
    class VkErrorDeviceLost(Exception):
        pass

    context = object.__new__(VulkanContext)
    context._device_lost = False
    context._device_lost_error = None
    context._submit_on_unchecked = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        VkErrorDeviceLost()
    )

    with pytest.raises(VkErrorDeviceLost):
        context.submit_on("graphics", lambda _command: None)

    assert context.device_lost is True
    assert context.device_lost_error == "VkErrorDeviceLost"


def test_submit_frame_waits_for_stereo_binary_semaphores_together() -> None:
    submissions = []

    class FakeVk:
        VK_STRUCTURE_TYPE_SUBMIT_INFO = 1
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 2
        VkSubmitInfo = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))

        @staticmethod
        def vkQueueSubmit(queue, count, infos, fence):
            submissions.append((queue, count, infos, fence))

    context = object.__new__(VulkanContext)
    context.vk = FakeVk()
    context._timeline_semaphore = None

    context._submit_frame(
        queue="graphics",
        command_buffer="commands",
        fence="fence",
        timeline_value=1,
        wait_semaphore=("left-finished", "right-finished"),
    )

    submit = submissions[0][2][0]
    assert submit.waitSemaphoreCount == 2
    assert submit.pWaitSemaphores == ["left-finished", "right-finished"]
