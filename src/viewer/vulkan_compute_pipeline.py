from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from viewer.vulkan_descriptors import DescriptorBinding, create_descriptor_set_layout


SPIRV_MAGIC = 0x07230203


class VulkanComputePipelineError(RuntimeError):
    pass


def read_spirv_words(path: str | Path) -> list[int]:
    shader_path = Path(path)
    try:
        payload = shader_path.read_bytes()
    except OSError as exc:
        raise VulkanComputePipelineError(
            f"unable to read SPIR-V shader: {shader_path}"
        ) from exc
    if not payload or len(payload) % 4:
        raise VulkanComputePipelineError("SPIR-V payload must be a non-empty 32-bit word array")
    words = list(struct.unpack(f"<{len(payload) // 4}I", payload))
    if words[0] != SPIRV_MAGIC:
        raise VulkanComputePipelineError(f"invalid SPIR-V magic in {shader_path}")
    return words


class VulkanComputePipeline:
    def __init__(
        self,
        context: Any,
        shader_path: str | Path,
        *,
        descriptor_bindings: list[DescriptorBinding] | None = None,
    ) -> None:
        self.context = context
        self.vk = context.vk
        self.shader_module = None
        self.descriptor_set_layout = None
        self.pipeline_layout = None
        self.pipeline = None
        words = read_spirv_words(shader_path)
        payload = struct.pack(f"<{len(words)}I", *words)
        try:
            self.shader_module = self.vk.vkCreateShaderModule(
                context.device,
                self.vk.VkShaderModuleCreateInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    codeSize=len(words) * 4,
                    pCode=payload,
                ),
                None,
            )
            bindings = list(descriptor_bindings or [])
            if bindings:
                self.descriptor_set_layout = create_descriptor_set_layout(context, bindings)
            self.pipeline_layout = self.vk.vkCreatePipelineLayout(
                context.device,
                self.vk.VkPipelineLayoutCreateInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    setLayoutCount=1 if self.descriptor_set_layout is not None else 0,
                    pSetLayouts=[self.descriptor_set_layout]
                    if self.descriptor_set_layout is not None
                    else None,
                ),
                None,
            )
            stage = self.vk.VkPipelineShaderStageCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
                module=self.shader_module,
                pName="main",
            )
            self.pipeline = self.vk.vkCreateComputePipelines(
                context.device,
                None,
                1,
                [
                    self.vk.VkComputePipelineCreateInfo(
                        sType=self.vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                        stage=stage,
                        layout=self.pipeline_layout,
                    )
                ],
                None,
            )[0]
        except Exception:
            self.close()
            raise

    def record_dispatch(
        self,
        command_buffer: Any,
        *,
        group_count_x: int = 1,
        group_count_y: int = 1,
        group_count_z: int = 1,
        descriptor_set: Any | None = None,
    ) -> None:
        if self.pipeline is None:
            raise VulkanComputePipelineError("compute pipeline is closed")
        counts = (group_count_x, group_count_y, group_count_z)
        if any(int(value) < 1 for value in counts):
            raise ValueError("compute dispatch group counts must be positive")
        self.vk.vkCmdBindPipeline(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline,
        )
        if descriptor_set is not None:
            if self.descriptor_set_layout is None:
                raise VulkanComputePipelineError("descriptor set supplied to pipeline without a layout")
            self.vk.vkCmdBindDescriptorSets(
                command_buffer,
                self.vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline_layout,
                0,
                1,
                [descriptor_set],
                0,
                None,
            )
        self.vk.vkCmdDispatch(command_buffer, *(int(value) for value in counts))

    def close(self) -> None:
        if self.context.device is not None:
            if self.pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.pipeline, None)
            if self.pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.pipeline_layout, None
                )
            if self.descriptor_set_layout is not None:
                self.vk.vkDestroyDescriptorSetLayout(
                    self.context.device, self.descriptor_set_layout, None
                )
            if self.shader_module is not None:
                self.vk.vkDestroyShaderModule(
                    self.context.device, self.shader_module, None
                )
        self.pipeline = None
        self.pipeline_layout = None
        self.descriptor_set_layout = None
        self.shader_module = None

    def __enter__(self) -> "VulkanComputePipeline":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
