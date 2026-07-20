from __future__ import annotations

from pathlib import Path
from typing import Any

from stereo_runtime.vulkan_graph import (
    VulkanComputeGraph,
    VulkanComputePass,
    VulkanPassDeclaration,
    VulkanStereoSubmission,
)
from viewer.vulkan_compute_pipeline import VulkanComputePipeline
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    DescriptorBudget,
    VulkanDescriptorArena,
    VulkanStorageImage,
)


class VulkanImageCopyPass:
    """Bounded compute pass that copies one storage image into another."""

    WORKGROUP_SIZE = 8

    def __init__(
        self,
        context: Any,
        *,
        width: int,
        height: int,
        shader_path: str | Path = "shaders/d2s_copy_image.spv",
    ) -> None:
        if int(width) < 1 or int(height) < 1:
            raise ValueError("image pass dimensions must be positive")
        self.context = context
        self.width = int(width)
        self.height = int(height)
        self.pipeline: VulkanComputePipeline | None = None
        self.descriptor_arena: VulkanDescriptorArena | None = None
        self.descriptor_set: Any | None = None
        self.graph: VulkanComputeGraph | None = None
        try:
            storage_image = context.vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            self.pipeline = VulkanComputePipeline(
                context,
                shader_path,
                descriptor_bindings=[
                    DescriptorBinding(binding=0, descriptor_type=storage_image),
                    DescriptorBinding(binding=1, descriptor_type=storage_image),
                ],
            )
            self.descriptor_arena = VulkanDescriptorArena(
                context,
                DescriptorBudget(max_sets=1, storage_images_per_set=2),
            )
            self.descriptor_set = self.descriptor_arena.allocate(
                self.pipeline.descriptor_set_layout
            )
            self.graph = VulkanComputeGraph.from_passes(
                context,
                (
                    VulkanComputePass(
                        VulkanPassDeclaration(
                            name="copy_image",
                            group_counts=self.group_counts,
                            reads=("source_image",),
                            writes=("output_image",),
                        ),
                        self.pipeline,
                        self.descriptor_set,
                    ),
                ),
            )
        except Exception:
            self.close()
            raise

    @property
    def group_counts(self) -> tuple[int, int, int]:
        return (
            (self.width + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE,
            (self.height + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE,
            1,
        )

    def submit(
        self,
        source_image: VulkanStorageImage,
        output_image: VulkanStorageImage,
        *,
        frame_id: int,
        config_version: int,
        ready_timeline: int | None = None,
    ) -> int | None:
        if self.graph is None or self.descriptor_arena is None:
            raise RuntimeError("Vulkan image copy pass is closed")
        if source_image.image is output_image.image:
            raise RuntimeError("source and output storage images must be distinct")
        for image in (source_image, output_image):
            if image.context is not self.context:
                raise RuntimeError("storage image belongs to a different Vulkan context")
            if image.width != self.width or image.height != self.height:
                raise ValueError("image dimensions do not match the copy pass")
            state = self.context.image_state(image.image)
            if state.layout != self.context.vk.VK_IMAGE_LAYOUT_GENERAL:
                raise RuntimeError("storage image must be in GENERAL layout before dispatch")
            if state.queue_family_index != self.context.compute_queue_family_index:
                raise RuntimeError("storage image is not owned by the compute queue")
        self.descriptor_arena.update_storage_image(
            self.descriptor_set, 0, source_image
        )
        self.descriptor_arena.update_storage_image(
            self.descriptor_set, 1, output_image
        )
        return self.graph.submit(
            VulkanStereoSubmission(
                frame_id=frame_id,
                rgb_handle=source_image.image,
                depth_handle=output_image.image,
                config_version=config_version,
                ready_timeline=ready_timeline,
            )
        )

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
        if self.descriptor_arena is not None:
            self.descriptor_arena.close()
        if self.pipeline is not None:
            self.pipeline.close()
        self.graph = None
        self.descriptor_arena = None
        self.descriptor_set = None
        self.pipeline = None

    def __enter__(self) -> "VulkanImageCopyPass":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
