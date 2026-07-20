from __future__ import annotations

import pytest

from viewer.vulkan_resources import VulkanExternalImageRegistry, VulkanImageResource


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
