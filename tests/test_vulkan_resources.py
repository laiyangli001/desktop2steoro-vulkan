from __future__ import annotations

import os

import pytest

from viewer.vulkan_resources import (
    VulkanExternalImageRegistry,
    VulkanExternalMemoryContract,
    VulkanD3D11ImportedImage,
    VulkanImageResource,
)


class FakeContext:
    def __init__(self):
        self.states = {}

    def register_image_state(self, image, state):
        self.states[image] = state

    def unregister_image_state(self, image):
        self.states.pop(image, None)


def _resource(context, image="image"):
    return VulkanImageResource(
        context=context,
        image=image,
        view="view",
        width=1280,
        height=720,
        format=37,
        layout=1,
        access_mask=2,
        stage_mask=4,
        queue_family_index=0,
    )


def test_external_image_registry_registers_state_without_owning_handles():
    context = FakeContext()
    registry = VulkanExternalImageRegistry(context)
    resource = _resource(context)

    registry.register(resource)
    assert registry.registered_count == 1
    assert context.states["image"].layout == 1
    registry.unregister(resource)
    assert registry.registered_count == 0
    assert context.states == {}


def test_external_image_registry_rejects_duplicate_and_cross_context_resources():
    context = FakeContext()
    registry = VulkanExternalImageRegistry(context)
    resource = _resource(context)
    registry.register(resource)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(resource)
    with pytest.raises(ValueError, match="different Vulkan context"):
        registry.register(_resource(FakeContext(), image="other"))


def test_external_image_registry_discard_releases_non_owning_references():
    context = FakeContext()
    registry = VulkanExternalImageRegistry(context)
    registry.register(_resource(context))
    registry.discard()
    assert registry.registered_count == 0


def test_external_memory_contract_requires_explicit_handle_format_adapter_and_sync():
    contract = VulkanExternalMemoryContract(
        memory_handle=11,
        memory_handle_type=8,
        width=1280,
        height=720,
        format=44,
        allocation_size=4096,
        adapter_luid=99,
        ready_semaphore_handle=12,
        ready_semaphore_handle_type=3,
        producer_ready=True,
    )
    assert contract.is_d3d11_zero_copy_ready(
        windows=True, bgra8_format=44, d3d11_texture_handle_type=8
    )
    assert contract.validation_reasons(
        windows=True, bgra8_format=37, d3d11_texture_handle_type=8
    ) == (
        "format_not_d3d11_bgra8",
    )


def test_external_memory_contract_does_not_infer_zero_copy_from_memory_handle():
    contract = VulkanExternalMemoryContract(
        memory_handle=11,
        memory_handle_type=8,
        width=1280,
        height=720,
        format=44,
        adapter_luid=99,
    )
    assert not contract.is_d3d11_zero_copy_ready(
        windows=True, bgra8_format=44, d3d11_texture_handle_type=8
    )
    assert "missing_producer_ready_sync" in contract.validation_reasons(
        windows=True, bgra8_format=44, d3d11_texture_handle_type=8
    )


def test_external_memory_contract_rejects_opaque_win32_for_d3d11_texture_import():
    contract = VulkanExternalMemoryContract(
        memory_handle=11,
        memory_handle_type=2,
        width=1280,
        height=720,
        format=44,
        adapter_luid=99,
        ready_semaphore_handle=12,
        ready_semaphore_handle_type=3,
        producer_ready=True,
    )
    assert not contract.is_d3d11_zero_copy_ready(
        windows=True, bgra8_format=44, d3d11_texture_handle_type=8
    )
    assert "handle_type_not_d3d11_texture" in contract.validation_reasons(
        windows=True, bgra8_format=44, d3d11_texture_handle_type=8
    )


@pytest.mark.skipif(os.name != "nt", reason="D3D11/Vulkan shared import is Windows-only")
def test_d3d11_vulkan_import_requires_a_verified_vulkan_adapter_luid():
    with pytest.raises(RuntimeError, match="Vulkan adapter LUID is unavailable"):
        VulkanD3D11ImportedImage(
            object(),
            1280,
            720,
            123,
            adapter_luid=456,
        )
