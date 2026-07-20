from __future__ import annotations

from pathlib import Path

import pytest

from app_runtime.probe import build_capability_report
from stereo_runtime.vulkan_graph import (
    VulkanComputeGraph,
    VulkanComputePass,
    VulkanGraphState,
    VulkanPassDeclaration,
    VulkanStereoSubmission,
)
from viewer.vulkan_compute_pipeline import (
    VulkanComputePipelineError,
    read_spirv_words,
)
from viewer.vulkan_descriptors import DescriptorBinding, DescriptorBudget


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_capability_report_identifies_new_project():
    report = build_capability_report()
    assert report["project"] == "desktop2steoro-vulkan"
    assert report["migration"]["python_vulkan_runtime"] == "phase1_implemented"
    assert report["migration"]["openxr_vulkan_session"] == "phase1_validated"


def test_current_style_source_layout_is_present():
    expected = [
        "src/main.py",
        "src/app_runtime",
        "src/capture",
        "src/gui",
        "src/stereo_runtime",
        "src/viewer",
        "src/xr_viewer",
        "native/filament/bridge/CMakeLists.txt",
        ".github/workflows/filament-bridge.yml",
    ]
    for relative in expected:
        assert (PROJECT_ROOT / relative).exists(), relative


def test_forbidden_legacy_runtime_directories_are_not_migrated():
    forbidden = [
        "src/xr_viewer/panda_runtime",
        "src/capture/dxgi/native",
    ]
    for relative in forbidden:
        assert not (PROJECT_ROOT / relative).exists(), relative


def test_vulkan_submission_contract_is_python_native():
    submission = VulkanStereoSubmission(
        frame_id=7,
        rgb_handle=object(),
        depth_handle=object(),
        config_version=3,
    )
    assert submission.frame_id == 7
    assert VulkanGraphState.CREATED.value == "created"


def test_vulkan_compute_graph_submits_latest_frame_to_compute_queue():
    calls = []

    class FakeContext:
        def submit_on(self, role, record):
            command_buffer = object()
            record(command_buffer)
            calls.append(role)
            return 12

    def record_pass(command_buffer, submission):
        calls.append((command_buffer, submission.frame_id))

    graph = VulkanComputeGraph(FakeContext(), record_pass)
    graph.enqueue(VulkanStereoSubmission(1, object(), object(), 1))
    graph.enqueue(VulkanStereoSubmission(2, object(), object(), 1))
    assert graph.flush() == 12
    assert calls[0][1] == 2
    assert calls[-1] == "compute"


def test_vulkan_compute_graph_from_pipeline_records_dispatch():
    calls = []

    class FakeContext:
        def submit_on(self, role, record):
            record("command-buffer")
            calls.append(role)
            return 4

    class FakePipeline:
        def record_dispatch(self, command_buffer, **counts):
            calls.append((command_buffer, counts))

    graph = VulkanComputeGraph.from_pipeline(
        FakeContext(), FakePipeline(), group_counts=(2, 3, 1)
    )
    assert graph.submit(VulkanStereoSubmission(1, object(), object(), 1)) == 4
    assert calls == [
        ("command-buffer", {
            "group_count_x": 2,
            "group_count_y": 3,
            "group_count_z": 1,
        }),
        "compute",
    ]


def test_vulkan_compute_graph_forwards_input_ready_timeline():
    calls = []

    class FakeContext:
        def submit_on(self, role, record, **kwargs):
            record("command-buffer")
            calls.append((role, kwargs))
            return 9

    graph = VulkanComputeGraph(FakeContext(), lambda *_: None)
    assert graph.submit(
        VulkanStereoSubmission(3, object(), object(), 1, ready_timeline=7)
    ) == 9
    assert calls == [("compute", {"wait_for_timeline": 7})]


def test_vulkan_compute_graph_from_passes_inserts_barrier_between_passes():
    calls = []

    class FakeVk:
        VK_STRUCTURE_TYPE_MEMORY_BARRIER = 1
        VK_ACCESS_SHADER_WRITE_BIT = 2
        VK_ACCESS_SHADER_READ_BIT = 4
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 8

        @staticmethod
        def VkMemoryBarrier(**kwargs):
            return kwargs

        @staticmethod
        def vkCmdPipelineBarrier(*args):
            calls.append(args)

    class FakeContext:
        vk = FakeVk()

        def submit_on(self, role, record, **kwargs):
            assert role == "compute"
            record("command-buffer")
            return 11

    class FakePipeline:
        def __init__(self, name):
            self.name = name

        def record_dispatch(self, command_buffer, **kwargs):
            calls.append((self.name, command_buffer, kwargs))

    graph = VulkanComputeGraph.from_passes(
        FakeContext(),
        (
            VulkanComputePass(
                VulkanPassDeclaration("normalize", writes=("normalized",)),
                FakePipeline("a"),
            ),
            VulkanComputePass(VulkanPassDeclaration("warp", reads=("normalized",)), FakePipeline("b")),
        ),
    )
    assert graph.submit(VulkanStereoSubmission(1, object(), object(), 1)) == 11
    assert calls[0][0] == "a"
    assert calls[-1][0] == "b"
    assert any(call[0] == "command-buffer" for call in calls[1:-1])


def test_vulkan_compute_graph_close_rejects_new_work():
    graph = VulkanComputeGraph(type("Context", (), {})(), lambda *_: None)
    graph.close()
    with pytest.raises(RuntimeError, match="not ready"):
        graph.enqueue(VulkanStereoSubmission(1, object(), object(), 1))


def test_spirv_loader_validates_magic_and_word_alignment(tmp_path):
    shader = tmp_path / "noop.spv"
    shader.write_bytes((0x07230203).to_bytes(4, "little") + b"\0" * 4)
    assert read_spirv_words(shader) == [0x07230203, 0]

    invalid = tmp_path / "invalid.spv"
    invalid.write_bytes(b"bad")
    with pytest.raises(VulkanComputePipelineError, match="32-bit"):
        read_spirv_words(invalid)


def test_descriptor_budget_is_bounded():
    budget = DescriptorBudget(max_sets=4, storage_buffers_per_set=2)
    assert budget.max_sets == 4
    with pytest.raises(ValueError, match="at least one"):
        DescriptorBudget(max_sets=0)


def test_descriptor_binding_requires_positive_count():
    binding = DescriptorBinding(binding=0, descriptor_type=7)
    assert binding.descriptor_count == 1
    with pytest.raises(ValueError, match="positive"):
        DescriptorBinding(binding=0, descriptor_type=7, descriptor_count=0)
