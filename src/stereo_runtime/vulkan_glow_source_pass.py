from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from viewer.vulkan_compute_pipeline import VulkanComputePipeline
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    DescriptorBudget,
    VulkanDescriptorArena,
)


class VulkanGlowSourcePass:
    """Prefilter planar sRGB RGB into a small linear RGBA Vulkan image."""

    WORKGROUP_SIZE = 8
    PUSH_CONSTANTS_SIZE = 32

    def __init__(
        self,
        context: Any,
        *,
        target_width: int = 320,
        target_height: int = 180,
        slot_count: int = 3,
        shader_path: str | Path | None = None,
    ) -> None:
        self.context = context
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.slot_count = max(2, int(slot_count))
        if self.target_width < 1 or self.target_height < 1:
            raise ValueError("Glow target dimensions must be positive")
        shader = Path(shader_path) if shader_path is not None else (
            Path(__file__).resolve().parents[2] / "shaders" / "d2s_glow_source.spv"
        )
        vk = context.vk
        self.pipeline = VulkanComputePipeline(
            context,
            shader,
            descriptor_bindings=[
                DescriptorBinding(
                    binding=0, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                ),
                DescriptorBinding(
                    binding=1, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                ),
                DescriptorBinding(
                    binding=2, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                ),
                DescriptorBinding(
                    binding=3, descriptor_type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                ),
            ],
            push_constants_size=self.PUSH_CONSTANTS_SIZE,
        )
        self.descriptor_arena = VulkanDescriptorArena(
            context,
            DescriptorBudget(
                max_sets=self.slot_count,
                storage_buffers_per_set=3,
                storage_images_per_set=1,
            ),
        )
        self.descriptor_sets = [
            self.descriptor_arena.allocate(self.pipeline.descriptor_set_layout)
            for _ in range(self.slot_count)
        ]

    @property
    def group_counts(self) -> tuple[int, int, int]:
        return (
            (self.target_width + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE,
            (self.target_height + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE,
            1,
        )

    @staticmethod
    def input_buffer_size(source_width: int, source_height: int) -> int:
        return int(source_width) * int(source_height) * 3 * 4

    def record(
        self,
        command_buffer: Any,
        *,
        slot_index: int,
        source_buffer: Any,
        output_image: Any,
        screen_light_buffer: Any,
        history_buffer: Any,
        source_width: int,
        source_height: int,
        prefilter_scale: float,
        surround_region_average: bool = False,
        screen_light_only: bool = False,
        temporal_alpha: float = 1.0,
    ) -> None:
        if source_buffer.context is not self.context or output_image.context is not self.context:
            raise ValueError("Glow resources belong to a different Vulkan context")
        if screen_light_buffer.context is not self.context:
            raise ValueError("screen-light buffer belongs to a different Vulkan context")
        if int(screen_light_buffer.size) < 16:
            raise ValueError("screen-light buffer must contain one vec4")
        if history_buffer.context is not self.context:
            raise ValueError("Glow history buffer belongs to a different Vulkan context")
        if int(history_buffer.size) < self.target_width * self.target_height * 16:
            raise ValueError("Glow history buffer must contain one vec4 per output pixel")
        if int(source_buffer.size) < self.input_buffer_size(source_width, source_height):
            raise ValueError("Glow source buffer is too small")
        if (
            int(output_image.width) != self.target_width
            or int(output_image.height) != self.target_height
        ):
            raise ValueError("Glow output dimensions do not match the pass")
        state = self.context.image_state(output_image.image)
        if state.layout != self.context.vk.VK_IMAGE_LAYOUT_GENERAL:
            raise ValueError("Glow output image must be in GENERAL layout")
        descriptor_set = self.descriptor_sets[int(slot_index) % len(self.descriptor_sets)]
        self.descriptor_arena.update_storage_buffer(descriptor_set, 0, source_buffer)
        self.descriptor_arena.update_storage_image(descriptor_set, 1, output_image)
        self.descriptor_arena.update_storage_buffer(
            descriptor_set, 2, screen_light_buffer
        )
        self.descriptor_arena.update_storage_buffer(descriptor_set, 3, history_buffer)
        push_constants = struct.pack(
            "<IIIIfIfI",
            int(source_width),
            int(source_height),
            self.target_width,
            self.target_height,
            max(1.0, float(prefilter_scale)),
            int(bool(surround_region_average)),
            max(0.0, min(1.0, float(temporal_alpha))),
            int(not screen_light_only),
        )
        group_counts = (1, 1, 1) if screen_light_only else self.group_counts
        self.pipeline.record_dispatch(
            command_buffer,
            group_count_x=group_counts[0],
            group_count_y=group_counts[1],
            group_count_z=group_counts[2],
            descriptor_set=descriptor_set,
            push_constants=push_constants,
        )

    def close(self) -> None:
        self.descriptor_arena.close()
        self.pipeline.close()
        self.descriptor_sets = []
