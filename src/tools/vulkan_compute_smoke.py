from __future__ import annotations

import struct
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
from stereo_runtime.vulkan_stereo_pass import VulkanStereoFusedParams, VulkanStereoFusedPass


def main() -> int:
    with VulkanContext.create() as context:
        binding = DescriptorBinding(
            binding=0,
            descriptor_type=context.vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        )
        with VulkanComputePipeline(
            context,
            "src/shaders/d2s_storage_increment.spv",
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
        width, height = 32, 24
        pixels = width * height
        buffer_sizes = {
            "rgb": pixels * 3 * 4,
            "depth": pixels * 4,
            "left_eye": pixels * 3 * 4,
            "right_eye": pixels * 3 * 4,
            "mask": pixels * 4,
        }
        with (
            VulkanStereoFusedPass(context, width=width, height=height) as stereo_pass,
            VulkanStorageBuffer(context, buffer_sizes["rgb"]) as rgb,
            VulkanStorageBuffer(context, buffer_sizes["depth"]) as depth,
            VulkanStorageBuffer(context, buffer_sizes["left_eye"]) as left_eye,
            VulkanStorageBuffer(context, buffer_sizes["right_eye"]) as right_eye,
            VulkanStorageBuffer(context, buffer_sizes["mask"]) as mask,
        ):
            rgb.write_bytes(struct.pack("<f", 0.25) * (pixels * 3))
            depth.write_bytes(struct.pack("<f", 0.5) * pixels)
            stereo_pass.submit(
                rgb,
                depth,
                left_eye,
                right_eye,
                mask,
                params=VulkanStereoFusedParams(max_disparity_px=8.0),
                frame_id=3,
                config_version=1,
            )
            context.wait_idle()
            left_value = struct.unpack("<f", left_eye.read_bytes(4))[0]
            right_value = struct.unpack("<f", right_eye.read_bytes(4))[0]
            mask_value = struct.unpack("<f", mask.read_bytes(4))[0]
            if not (0.0 <= left_value <= 1.0 and 0.0 <= right_value <= 1.0):
                raise RuntimeError("fused Vulkan stereo output is outside the RGB range")
            if mask_value != 0.0:
                raise RuntimeError("uniform-depth fused Vulkan stereo mask should be zero")
            print(
                "vulkan_stereo_fused: PASS "
                f"groups={stereo_pass.group_counts} left={left_value:.4f} right={right_value:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
