from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_runtime.vulkan_runtime import VulkanRuntimeConfig, VulkanRuntimeSession
from viewer.vulkan_compute_pipeline import VulkanComputePipeline
from viewer.vulkan_context import VulkanContext
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    DescriptorBudget,
    VulkanDescriptorArena,
    VulkanStorageImage,
    VulkanStorageBuffer,
)
from stereo_runtime.vulkan_graph import VulkanComputeGraph, VulkanStereoSubmission


def main() -> int:
    with VulkanContext.create() as context:
        binding = DescriptorBinding(
            binding=0,
            descriptor_type=context.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        )
        with VulkanComputePipeline(
            context,
            "shaders/d2s_storage_increment.spv",
            descriptor_bindings=[binding],
        ) as pipeline, VulkanDescriptorArena(context) as arena, VulkanStorageBuffer(
            context, 4
        ) as storage:
            storage.write_uint32(41)
            descriptor_set = arena.allocate(pipeline.descriptor_set_layout)
            arena.update_storage_buffer(descriptor_set, 0, storage)
            graph = VulkanComputeGraph.from_pipeline(
                context,
                pipeline,
                descriptor_set=descriptor_set,
            )
            timeline_value = graph.submit(
                VulkanStereoSubmission(
                    frame_id=1,
                    rgb_handle=object(),
                    depth_handle=object(),
                    config_version=1,
                )
            )
            context.wait_idle()
            if storage.read_uint32() != 42:
                raise RuntimeError("storage buffer dispatch produced an unexpected value")
            print(
                "vulkan_compute_smoke: PASS "
                f"timeline={timeline_value} state={graph.state.value}"
            )
            graph.close()
        with VulkanRuntimeSession(
            context, VulkanRuntimeConfig(width=1, height=1)
        ) as runtime, VulkanStorageImage(context) as source_image, VulkanStorageImage(
            context
        ) as output_image:
            source_ready = source_image.transition_to_general()
            output_ready = output_image.transition_to_general()
            runtime.submit_image_pair(
                source_image,
                output_image,
                frame_id=2,
                config_version=1,
                ready_timeline=max(source_ready, output_ready),
            )
            context.wait_idle()
            print("storage_image_dispatch: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
