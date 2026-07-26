from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

import stereo_runtime.vulkan_stereo_pass as module
from stereo_runtime.vulkan_stereo_pass import VulkanStereoFusedParams, VulkanStereoFusedPass


def test_vulkan_stereo_params_match_shader_push_constant_layout():
    payload = VulkanStereoFusedParams(
        depth_strength=2.0,
        max_disparity_px=96.0,
        convergence=0.1,
        edge_threshold=0.03,
        fill_strength=0.6,
        fill_radius=1,
        mask_feather_radius=3,
        symmetric=True,
    ).pack(3840, 2160)

    assert len(payload) == VulkanStereoFusedPass.PUSH_CONSTANTS_SIZE
    unpacked = struct.unpack("<IIfffffIII", payload)
    assert unpacked[:2] == (3840, 2160)
    assert unpacked[-3:] == (1, 3, 1)


def test_vulkan_stereo_params_reject_invalid_dimensions_and_radius():
    with pytest.raises(ValueError, match="dimensions"):
        VulkanStereoFusedParams().pack(0, 1)
    with pytest.raises(ValueError, match="radii"):
        VulkanStereoFusedParams(fill_radius=-1).pack(1, 1)


def test_vulkan_stereo_pass_uses_bounded_descriptor_sets(monkeypatch):
    class FakeVk:
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7

    class FakePipeline:
        def __init__(self, context, shader_path, **kwargs):
            self.descriptor_set_layout = "layout"
            self.calls = []

        def record_dispatch(self, command_buffer, **kwargs):
            self.calls.append((command_buffer, kwargs))

        def close(self):
            pass

    class FakeArena:
        def __init__(self, context, budget):
            self.sets = []

        def allocate(self, layout):
            value = f"set-{len(self.sets)}"
            self.sets.append(value)
            return value

        def update_storage_buffer(self, descriptor_set, binding, buffer):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "VulkanComputePipeline", FakePipeline)
    monkeypatch.setattr(module, "VulkanDescriptorArena", FakeArena)
    context = SimpleNamespace(vk=FakeVk(), frame_context_count=3)
    stereo_pass = VulkanStereoFusedPass(context, width=32, height=24)

    sizes = stereo_pass.buffer_sizes
    buffers = tuple(
        SimpleNamespace(context=context, size=sizes[name])
        for name in sizes
    )
    recorded = []

    def submit_on(role, record, **kwargs):
        assert role == "compute"
        record("command-buffer")
        recorded.append(kwargs)
        return 5

    context.submit_on = submit_on
    assert stereo_pass.submit(*buffers, frame_id=1, config_version=1) == 5
    assert len(stereo_pass.descriptor_sets) == 3
    assert recorded == [{}]
    assert stereo_pass.pipeline.calls[0][1]["group_count_x"] == 2
    assert stereo_pass.pipeline.calls[0][1]["group_count_y"] == 2
    assert len(stereo_pass.pipeline.calls[0][1]["push_constants"]) == 40
    stereo_pass.close()
