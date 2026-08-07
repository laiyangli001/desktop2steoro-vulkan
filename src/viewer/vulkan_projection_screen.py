from __future__ import annotations

import os
from pathlib import Path
import math
import struct
from typing import Any

from viewer.vulkan_compute_pipeline import read_spirv_words
from viewer.vulkan_context import ImageState
from viewer.vulkan_descriptors import DescriptorBinding, create_descriptor_set_layout
from viewer.vulkan_resources import VulkanTransientImage


class VulkanProjectionScreenPass:
    """Rasterize the world-space screen directly into one projection layer."""

    _SEGMENTS = 48
    _VERTEX_COUNT = (_SEGMENTS + 1) * 2
    _PUSH_CONSTANT_SIZE = 128
    _DESCRIPTOR_COUNT = 6
    _QUALITY_SLOT_COUNT = 3
    _DEFAULT_MIP_LOD_BIAS = -0.35
    _DEFAULT_MAX_MIP_LOD = 0.35
    _DEFAULT_RCAS_SHARPNESS = 0.5

    def __init__(self, context: Any, target_format: int) -> None:
        self.context = context
        self.vk = context.vk
        self.target_format = int(target_format)
        self.min_mip_lod = self._min_mip_lod_from_env()
        self.mip_lod_bias = self._mip_lod_bias_from_env()
        self.max_mip_lod = self._max_mip_lod_from_env()
        self.rcas_sharpness = self._rcas_sharpness_from_env()
        (
            self.min_mip_lod,
            self.max_mip_lod,
            self.mip_lod_bias,
            self.rcas_sharpness,
        ) = self._normalize_sampling_config(
            self.min_mip_lod,
            self.max_mip_lod,
            self.mip_lod_bias,
            self.rcas_sharpness,
        )
        self.shader_modules: list[Any] = []
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_sets: list[Any] = []
        self.descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.rcas_descriptor_sets: list[Any] = []
        self.rcas_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.quality_descriptor_sets: list[Any] = []
        self.quality_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.glow_descriptor_sets: list[Any] = []
        self.glow_descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.quality_slot_timelines = [0] * self._QUALITY_SLOT_COUNT
        self.quality_images: dict[tuple[int, int, int], list[VulkanTransientImage]] = {}
        self.mip_slot_timelines = [0] * self._QUALITY_SLOT_COUNT
        self.mip_images: dict[tuple[int, int, int], list[VulkanTransientImage]] = {}
        self.last_submit_profile: dict[str, float] = {}
        self.sampler = None
        self._retired_samplers: list[tuple[Any, int]] = []
        self._last_submit_timeline = 0
        self.render_pass = None
        self.overlay_render_pass = None
        self.pipeline_layout = None
        self.pipeline = None
        self.rcas_pipeline_layout = None
        self.rcas_pipeline = None
        self.copy_pipeline_layout = None
        self.copy_pipeline = None
        self.quality_pipeline_layout = None
        self.quality_pipeline = None
        self.image_views: dict[tuple[int, int], Any] = {}
        self.framebuffers: dict[tuple[int, int, int, int], Any] = {}
        self.overlay_framebuffers: dict[tuple[int, int, int, int], Any] = {}
        try:
            self._create()
        except Exception:
            self.close()
            raise

    @classmethod
    def _min_mip_lod_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_MIN_LOD", "")
        if not raw_value.strip():
            return 0.0
        try:
            return max(0.0, min(16.0, float(raw_value)))
        except ValueError:
            return 0.0

    @classmethod
    def _mip_lod_bias_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_MIP_LOD_BIAS", "")
        if not raw_value.strip():
            return cls._DEFAULT_MIP_LOD_BIAS
        try:
            return max(-1.5, min(0.0, float(raw_value)))
        except ValueError:
            return cls._DEFAULT_MIP_LOD_BIAS

    @classmethod
    def _rcas_sharpness_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_RCAS_SHARPNESS", "")
        if not raw_value.strip():
            return cls._DEFAULT_RCAS_SHARPNESS
        try:
            return max(0.0, min(1.0, float(raw_value)))
        except ValueError:
            return cls._DEFAULT_RCAS_SHARPNESS

    @classmethod
    def _max_mip_lod_from_env(cls) -> float:
        raw_value = os.environ.get("D2S_VULKAN_PROJECTION_MAX_LOD", "")
        if not raw_value.strip():
            return cls._DEFAULT_MAX_MIP_LOD
        try:
            return max(0.0, min(16.0, float(raw_value)))
        except ValueError:
            return cls._DEFAULT_MAX_MIP_LOD

    @staticmethod
    def _normalize_sampling_config(min_lod, max_lod, mip_lod_bias, rcas_sharpness) -> tuple[float, float, float, float]:
        minimum = max(0.0, min(16.0, float(min_lod)))
        maximum = max(0.0, min(16.0, float(max_lod)))
        if maximum < minimum:
            maximum = minimum
        bias = max(-1.5, min(0.0, float(mip_lod_bias)))
        sharpness = max(0.0, min(1.0, float(rcas_sharpness)))
        return minimum, maximum, bias, sharpness

    def set_sampling_config(self, *, min_lod, max_lod, mip_lod_bias, rcas_sharpness) -> bool:
        """Apply projection sampling on the presenter thread."""
        values = self._normalize_sampling_config(min_lod, max_lod, mip_lod_bias, rcas_sharpness)
        minimum, maximum, bias, sharpness = values
        changed = values != (self.min_mip_lod, self.max_mip_lod, self.mip_lod_bias, self.rcas_sharpness)
        self.rcas_sharpness = sharpness
        if not changed:
            self._collect_retired_samplers()
            return False
        if (minimum, maximum, bias) != (self.min_mip_lod, self.max_mip_lod, self.mip_lod_bias):
            old_sampler = self.sampler
            self.min_mip_lod, self.max_mip_lod, self.mip_lod_bias = minimum, maximum, bias
            self.sampler = self._create_sampler()
            if old_sampler is not None:
                self._retired_samplers.append((old_sampler, int(getattr(self, "_last_submit_timeline", 0))))
        self._collect_retired_samplers()
        return True

    def _create_sampler(self):
        vk = self.vk
        return vk.vkCreateSampler(
            self.context.device,
            vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=vk.VK_FILTER_LINEAR,
                minFilter=vk.VK_FILTER_LINEAR,
                mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
                addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                maxAnisotropy=1.0,
                mipLodBias=self.mip_lod_bias,
                minLod=self.min_mip_lod,
                maxLod=self.max_mip_lod,
            ),
            None,
        )

    def _collect_retired_samplers(self) -> None:
        completed = self.context.completed_timeline_value()
        if completed is None:
            return
        pending = []
        for sampler, timeline in self._retired_samplers:
            if int(timeline) <= int(completed):
                self.vk.vkDestroySampler(self.context.device, sampler, None)
            else:
                pending.append((sampler, timeline))
        self._retired_samplers = pending

    def _create_shader_module(self, path: Path) -> Any:
        words = read_spirv_words(path)
        payload = struct.pack(f"<{len(words)}I", *words)
        module = self.vk.vkCreateShaderModule(
            self.context.device,
            self.vk.VkShaderModuleCreateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(payload),
                pCode=payload,
            ),
            None,
        )
        self.shader_modules.append(module)
        return module

    def _create(self) -> None:
        vk = self.vk
        shader_root = Path(__file__).resolve().parents[2] / "shaders"
        vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_screen_vert.spv"
        )
        fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_screen_frag.spv"
        )
        rcas_vertex_module = self._create_shader_module(
            shader_root / "d2s_projection_rcas_vert.spv"
        )
        rcas_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_rcas_frag.spv"
        )
        copy_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_copy_frag.spv"
        )
        quality_fragment_module = self._create_shader_module(
            shader_root / "d2s_projection_quality_frag.spv"
        )
        self.descriptor_set_layout = create_descriptor_set_layout(
            self.context,
            [
                DescriptorBinding(
                    0,
                    vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    stage_flags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                )
            ],
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.context.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=self._DESCRIPTOR_COUNT * 4,
                poolSizeCount=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        descriptorCount=self._DESCRIPTOR_COUNT * 4,
                    )
                ],
            ),
            None,
        )
        allocated_descriptor_sets = list(
            vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.descriptor_pool,
                    descriptorSetCount=self._DESCRIPTOR_COUNT * 4,
                    pSetLayouts=[self.descriptor_set_layout] * (self._DESCRIPTOR_COUNT * 4),
                ),
            )
        )
        self.descriptor_sets = allocated_descriptor_sets[:self._DESCRIPTOR_COUNT]
        self.rcas_descriptor_sets = allocated_descriptor_sets[
            self._DESCRIPTOR_COUNT:self._DESCRIPTOR_COUNT * 2
        ]
        self.quality_descriptor_sets = allocated_descriptor_sets[self._DESCRIPTOR_COUNT * 2:]
        self.quality_descriptor_sets = self.quality_descriptor_sets[:self._DESCRIPTOR_COUNT]
        self.glow_descriptor_sets = allocated_descriptor_sets[self._DESCRIPTOR_COUNT * 3:]
        self.sampler = self._create_sampler()
        attachment = vk.VkAttachmentDescription(
            format=self.target_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        color_reference = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        self.render_pass = vk.vkCreateRenderPass(
            self.context.device,
            vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1,
                pAttachments=[attachment],
                subpassCount=1,
                pSubpasses=[
                    vk.VkSubpassDescription(
                        pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                        colorAttachmentCount=1,
                        pColorAttachments=[color_reference],
                    )
                ],
            ),
            None,
        )
        overlay_attachment = vk.VkAttachmentDescription(
            format=self.target_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_LOAD,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        self.overlay_render_pass = vk.vkCreateRenderPass(
            self.context.device,
            vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1, pAttachments=[overlay_attachment],
                subpassCount=1,
                pSubpasses=[vk.VkSubpassDescription(
                    pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    colorAttachmentCount=1, pColorAttachments=[color_reference],
                )],
            ),
            None,
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=(
                            vk.VK_SHADER_STAGE_VERTEX_BIT
                            | vk.VK_SHADER_STAGE_FRAGMENT_BIT
                        ),
                        offset=0,
                        size=self._PUSH_CONSTANT_SIZE,
                    ),
                ],
            ),
            None,
        )
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=fragment_module,
                pName="main",
            ),
        ]
        self.pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=len(stages),
                    pStages=stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_TRUE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.rcas_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        offset=0,
                        size=16,
                    )
                ],
            ),
            None,
        )
        rcas_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=rcas_fragment_module,
                pName="main",
            ),
        ]
        self.rcas_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=rcas_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_FALSE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.rcas_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.copy_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
            ),
            None,
        )
        copy_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=copy_fragment_module,
                pName="main",
            ),
        ]
        self.copy_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=copy_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[
                            vk.VkPipelineColorBlendAttachmentState(
                                blendEnable=vk.VK_FALSE,
                                colorWriteMask=(
                                    vk.VK_COLOR_COMPONENT_R_BIT
                                    | vk.VK_COLOR_COMPONENT_G_BIT
                                    | vk.VK_COLOR_COMPONENT_B_BIT
                                    | vk.VK_COLOR_COMPONENT_A_BIT
                                ),
                            )
                        ],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[
                            vk.VK_DYNAMIC_STATE_VIEWPORT,
                            vk.VK_DYNAMIC_STATE_SCISSOR,
                        ],
                    ),
                    layout=self.copy_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]
        self.quality_pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[vk.VkPushConstantRange(
                    stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    offset=0,
                    size=16,
                )],
            ),
            None,
        )
        quality_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=rcas_vertex_module,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=quality_fragment_module,
                pName="main",
            ),
        ]
        self.quality_pipeline = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            1,
            [
                vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2,
                    pStages=quality_stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1,
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_NONE,
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=1,
                        pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                            blendEnable=vk.VK_FALSE,
                            colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT),
                        )],
                    ),
                    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                        dynamicStateCount=2,
                        pDynamicStates=[vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR],
                    ),
                    layout=self.quality_pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]

    def _target_view_and_framebuffer(
        self, target: Any, array_layer: int, *, overlay: bool = False
    ) -> tuple[Any, Any]:
        vk = self.vk
        view_key = (id(target.image), int(array_layer))
        view = self.image_views.get(view_key)
        if view is None:
            view = vk.vkCreateImageView(
                self.context.device,
                vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=target.image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=self.target_format,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=int(array_layer),
                        layerCount=1,
                    ),
                ),
                None,
            )
            self.image_views[view_key] = view
        framebuffer_key = (
            view_key[0], view_key[1], int(target.width), int(target.height)
        )
        framebuffer_cache = self.overlay_framebuffers if overlay else self.framebuffers
        framebuffer = framebuffer_cache.get(framebuffer_key)
        if framebuffer is None:
            framebuffer = vk.vkCreateFramebuffer(
                self.context.device,
                vk.VkFramebufferCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    renderPass=self.overlay_render_pass if overlay else self.render_pass,
                    attachmentCount=1,
                    pAttachments=[view],
                    width=int(target.width),
                    height=int(target.height),
                    layers=1,
                ),
                None,
            )
            framebuffer_cache[framebuffer_key] = framebuffer
        return view, framebuffer

    def submit(
        self,
        source: Any,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        push_constants: bytes,
        clear_color: tuple[float, float, float, float],
        wait_semaphore: Any | None,
    ) -> int:
        draw = self._prepare_draw(
            source,
            target,
            array_layer=array_layer,
            eye_index=eye_index,
            frame_slot=frame_slot,
            push_constants=push_constants,
            clear_color=clear_color,
        )
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: self._record_draw(command_buffer, draw),
            wait_semaphore=wait_semaphore,
        )
        self._complete_draw(draw, timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_stereo(self, draws: list[dict[str, Any]]) -> int:
        """Submit both eye projection draws in one graphics queue submission."""
        if len(draws) != 2:
            raise ValueError("stereo projection requires exactly two draws")
        prepared = [
            self._prepare_draw(
                item["source"],
                item["target"],
                array_layer=int(item["array_layer"]),
                eye_index=int(item["eye_index"]),
                frame_slot=int(item["frame_slot"]),
                push_constants=item["push_constants"],
                clear_color=item["clear_color"],
            )
            for item in draws
        ]
        wait_semaphores = [item["wait_semaphore"] for item in draws]
        wait_semaphores = [item for item in wait_semaphores if item is not None]
        submit_profile: dict[str, float] = {}
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, draw) for draw in prepared
            ],
            wait_semaphore=wait_semaphores,
            on_submit_profile=submit_profile.update,
        )
        self.last_submit_profile = submit_profile
        for draw in prepared:
            self._complete_draw(draw, timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def submit_stereo_glow(
        self,
        draws: list[dict[str, Any]],
        *,
        wait_for_timeline: int,
    ) -> int:
        """Blend completed Vulkan Glow sources over the SBS projection pair."""
        if len(draws) != 2 or self.pipeline is None:
            return int(wait_for_timeline)
        prepared = []
        completed = self.context.completed_timeline_value()
        for item in draws:
            source = item.get("glow_source")
            if source is None:
                return int(wait_for_timeline)
            eye_index = int(item["eye_index"])
            descriptor_index = (eye_index * 3 + int(item["frame_slot"])) % self._DESCRIPTOR_COUNT
            if completed is not None and self.glow_descriptor_timelines[descriptor_index] > completed:
                return int(wait_for_timeline)
            if self.context.image_state(source.image).layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return int(wait_for_timeline)
            descriptor_set = self.glow_descriptor_sets[descriptor_index]
            self.vk.vkUpdateDescriptorSets(
                self.context.device, 1,
                [self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=0, descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[self.vk.VkDescriptorImageInfo(
                        sampler=self.sampler, imageView=source.require_view(),
                        imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                )], 0, None,
            )
            overlay = self._prepare_draw(
                source, item["target"], array_layer=int(item["array_layer"]),
                eye_index=eye_index, frame_slot=int(item["frame_slot"]),
                push_constants=bytes(
                    bytearray(item["push_constants"][:124])
                    + struct.pack("<f", -1.0)
                ),
                clear_color=item["clear_color"],
                descriptor_set=descriptor_set,
                overlay=True,
            )
            overlay["target_old_layout"] = self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            overlay["clear_target"] = False
            prepared.append(overlay)
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [self._record_draw(command_buffer, draw) for draw in prepared],
            wait_for_timeline=int(wait_for_timeline),
        )
        for draw in prepared:
            self.glow_descriptor_timelines[draw["descriptor_index"]] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def try_submit_stereo_rcas(self, draws: list[dict[str, Any]]) -> int | None:
        """Submit filter and RCAS in one queue submission, or skip without waiting."""
        if len(draws) != 2 or self.rcas_sharpness <= 0.0 or self.rcas_pipeline is None:
            return None
        frame_slot = int(draws[0]["frame_slot"]) % self._QUALITY_SLOT_COUNT
        completed = self.context.completed_timeline_value()
        if completed is None:
            return None
        descriptor_indices = [
            (int(item["eye_index"]) * 3 + int(item["frame_slot"]))
            % self._DESCRIPTOR_COUNT
            for item in draws
        ]
        in_use = [self.quality_slot_timelines[frame_slot]]
        in_use.extend(self.descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.rcas_descriptor_timelines[index] for index in descriptor_indices)
        if any(int(value) > int(completed) for value in in_use):
            return None
        target = draws[0]["target"]
        key = (int(target.width), int(target.height), int(target.format))
        images = self.quality_images.get(key)
        if images is None:
            images = [
                VulkanTransientImage(
                    self.context,
                    key[0],
                    key[1],
                    format=key[2],
                    label=f"projection-rcas-eye{eye}-slot{slot}",
                )
                for eye in range(2)
                for slot in range(self._QUALITY_SLOT_COUNT)
            ]
            self.quality_images[key] = images
        filtered_draws = []
        rcas_draws = []
        for item in draws:
            eye_index = int(item["eye_index"])
            scratch = images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            if scratch is None:
                return None
            filtered = self._prepare_draw(
                item["source"],
                scratch,
                array_layer=0,
                eye_index=eye_index,
                frame_slot=int(item["frame_slot"]),
                push_constants=item["push_constants"],
                clear_color=item["clear_color"],
            )
            filtered["target_old_layout"] = self.context.image_state(scratch.image).layout
            filtered_draws.append(filtered)
            rcas_draws.append(
                self._prepare_rcas_draw(
                    scratch,
                    item["target"],
                    array_layer=int(item["array_layer"]),
                    eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]),
                    sharpness=self.rcas_sharpness,
                )
            )
        wait_semaphores = [item["wait_semaphore"] for item in draws]
        wait_semaphores = [item for item in wait_semaphores if item is not None]
        submit_profile: dict[str, float] = {}
        timeline = self.context.submit_on(
            "graphics",
            lambda command_buffer: [
                self._record_draw(command_buffer, filtered_draws[index])
                or self._record_rcas_draw(command_buffer, rcas_draws[index])
                for index in range(2)
            ],
            wait_semaphore=wait_semaphores,
            on_submit_profile=submit_profile.update,
        )
        self.last_submit_profile = submit_profile
        for filtered, rcas in zip(filtered_draws, rcas_draws):
            self._complete_draw(filtered, timeline)
            self._complete_rcas_draw(rcas, timeline)
        self.quality_slot_timelines[frame_slot] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def try_submit_stereo_mip(self, draws: list[dict[str, Any]]) -> int | None:
        """Generate the final per-eye mip chain, with optional RCAS on mip zero."""
        use_rcas = self.rcas_sharpness > 0.0
        if (
            len(draws) != 2
            or self.copy_pipeline is None
            or (use_rcas and self.rcas_pipeline is None)
        ):
            return None
        frame_slot = int(draws[0]["frame_slot"]) % self._QUALITY_SLOT_COUNT
        completed = self.context.completed_timeline_value()
        if completed is None:
            return None
        descriptor_indices = [
            (int(item["eye_index"]) * 3 + int(item["frame_slot"]))
            % self._DESCRIPTOR_COUNT
            for item in draws
        ]
        in_use = [self.quality_slot_timelines[frame_slot], self.mip_slot_timelines[frame_slot]]
        in_use.extend(self.descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.rcas_descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.quality_descriptor_timelines[index] for index in descriptor_indices)
        if any(int(value) > int(completed) for value in in_use):
            return None
        source = draws[0]["source"]
        if any(
            int(item["source"].width) != int(source.width)
            or int(item["source"].height) != int(source.height)
            or int(item["target"].format) != int(draws[0]["target"].format)
            for item in draws[1:]
        ):
            return None
        if not self._supports_linear_blit(int(draws[0]["target"].format)):
            return None
        key = (int(source.width), int(source.height), int(draws[0]["target"].format))
        quality_images = None
        if use_rcas:
            quality_images = self.quality_images.get(key)
            if quality_images is None:
                quality_images = [
                    VulkanTransientImage(
                        self.context,
                        key[0],
                        key[1],
                        format=key[2],
                        label=f"projection-native-rcas-eye{eye}-slot{slot}",
                    )
                    for eye in range(2)
                    for slot in range(self._QUALITY_SLOT_COUNT)
                ]
                self.quality_images[key] = quality_images
        images = self.mip_images.get(key)
        if images is None:
            if self.mip_images:
                if any(int(value) > int(completed) for value in self.mip_slot_timelines):
                    return None
                for stale_images in self.mip_images.values():
                    for stale_image in stale_images:
                        stale_image.close()
                self.mip_images.clear()
            mip_levels = int(math.floor(math.log2(max(key[0], key[1])))) + 1
            images = [
                VulkanTransientImage(
                    self.context,
                    key[0],
                    key[1],
                    format=key[2],
                    label=f"projection-mip-eye{eye}-slot{slot}",
                    mip_levels=mip_levels,
                )
                for eye in range(2)
                for slot in range(self._QUALITY_SLOT_COUNT)
            ]
            self.mip_images[key] = images
        copy_draws = []
        rcas_draws = []
        screen_draws = []
        for item, descriptor_index in zip(draws, descriptor_indices):
            eye_index = int(item["eye_index"])
            mip = images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            if mip is None:
                return None
            copy_target = mip
            if use_rcas:
                quality = quality_images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
                if quality is None:
                    return None
                copy_target = quality
            copy_draw = self._prepare_copy_draw(item["source"], copy_target, descriptor_index)
            copy_draws.append(copy_draw)
            if use_rcas:
                rcas = self._prepare_rcas_draw(
                    copy_target,
                    mip,
                    array_layer=0,
                    eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]),
                    sharpness=self.rcas_sharpness,
                )
                rcas["target_old_layout"] = self.context.image_state(mip.image).layout
                rcas_draws.append(rcas)
            screen_draws.append(
                self._prepare_draw(
                    mip,
                    item["target"],
                    array_layer=int(item["array_layer"]),
                    eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]),
                    push_constants=item["push_constants"],
                    clear_color=item["clear_color"],
                    source_ready_in_submission=True,
                )
            )
        wait_semaphores = [item["wait_semaphore"] for item in draws]
        wait_semaphores = [item for item in wait_semaphores if item is not None]
        submit_profile: dict[str, float] = {}
        def record_stereo(command_buffer: Any) -> None:
            for index, copy_draw in enumerate(copy_draws):
                self._record_copy_draw(command_buffer, copy_draw)
                mip_draw = copy_draw
                if use_rcas:
                    mip_draw = rcas_draws[index]
                    self._record_rcas_draw(command_buffer, mip_draw)
                self._record_generate_mips(
                    command_buffer,
                    mip_draw["target"],
                    int(mip_draw["target_old_layout"]),
                )
                self._record_draw(command_buffer, screen_draws[index])

        timeline = self.context.submit_on(
            "graphics",
            record_stereo,
            wait_semaphore=wait_semaphores,
            on_submit_profile=submit_profile.update,
        )
        self.last_submit_profile = submit_profile
        if use_rcas:
            for copy_draw, rcas_draw, screen_draw in zip(copy_draws, rcas_draws, screen_draws):
                self._complete_quality_mip_draw(copy_draw, rcas_draw, screen_draw, timeline)
        else:
            for copy_draw, screen_draw in zip(copy_draws, screen_draws):
                self._complete_mip_draw(copy_draw, screen_draw, timeline)
        self.quality_slot_timelines[frame_slot] = int(timeline)
        self.mip_slot_timelines[frame_slot] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def try_submit_stereo_quality_mip(
        self,
        draws: list[dict[str, Any]],
        *,
        mode: str,
        filter_scale: float,
        upscale_scale: float,
    ) -> int | None:
        """Match Filament's quality chain before the final curved-screen draw."""
        if mode == "native_mip":
            return self.try_submit_stereo_mip(draws)
        use_rcas = self.rcas_sharpness > 0.0
        if (
            len(draws) != 2
            or mode not in {"downsample_lanczos_rcas", "upscale_easu"}
            or self.quality_pipeline is None
            or self.copy_pipeline is None
            or (use_rcas and self.rcas_pipeline is None)
        ):
            return None
        frame_slot = int(draws[0]["frame_slot"]) % self._QUALITY_SLOT_COUNT
        completed = self.context.completed_timeline_value()
        if completed is None:
            return None
        source = draws[0]["source"]
        target_format = int(draws[0]["target"].format)
        if any(
            int(item["source"].width) != int(source.width)
            or int(item["source"].height) != int(source.height)
            or int(item["target"].format) != target_format
            for item in draws[1:]
        ) or not self._supports_linear_blit(target_format):
            return None
        scale = float(upscale_scale) if mode == "upscale_easu" else 1.0 / float(filter_scale)
        quality_width = max(16, int(round(int(source.width) * scale)) & ~1)
        quality_height = max(16, int(round(int(source.height) * scale)) & ~1)
        key = (quality_width, quality_height, target_format)
        quality_images = self.quality_images.get(key)
        if quality_images is None:
            quality_images = [
                VulkanTransientImage(self.context, quality_width, quality_height,
                    format=target_format, label=f"projection-quality-eye{eye}-slot{slot}")
                for eye in range(2) for slot in range(self._QUALITY_SLOT_COUNT)
            ]
            self.quality_images[key] = quality_images
        mip_images = self.mip_images.get(key)
        if mip_images is None:
            mip_levels = int(math.floor(math.log2(max(quality_width, quality_height)))) + 1
            mip_images = [
                VulkanTransientImage(self.context, quality_width, quality_height,
                    format=target_format, label=f"projection-quality-mip-eye{eye}-slot{slot}",
                    mip_levels=mip_levels)
                for eye in range(2) for slot in range(self._QUALITY_SLOT_COUNT)
            ]
            self.mip_images[key] = mip_images
        descriptor_indices = [
            (int(item["eye_index"]) * 3 + int(item["frame_slot"])) % self._DESCRIPTOR_COUNT
            for item in draws
        ]
        in_use = [self.quality_slot_timelines[frame_slot], self.mip_slot_timelines[frame_slot]]
        in_use.extend(self.descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.rcas_descriptor_timelines[index] for index in descriptor_indices)
        in_use.extend(self.quality_descriptor_timelines[index] for index in descriptor_indices)
        if any(int(value) > int(completed) for value in in_use):
            return None
        quality_draws = []
        rcas_draws = []
        screen_draws = []
        for item, descriptor_index in zip(draws, descriptor_indices):
            eye_index = int(item["eye_index"])
            quality = quality_images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            mip = mip_images[eye_index * self._QUALITY_SLOT_COUNT + frame_slot].resource
            if quality is None or mip is None:
                return None
            quality_draws.append(self._prepare_quality_draw(
                item["source"], quality, descriptor_index, mode=mode,
                scale=float(upscale_scale) if mode == "upscale_easu" else 1.0,
            ))
            if use_rcas:
                rcas = self._prepare_rcas_draw(
                    quality, mip, array_layer=0, eye_index=eye_index,
                    frame_slot=int(item["frame_slot"]), sharpness=self.rcas_sharpness,
                )
                rcas["target_old_layout"] = self.context.image_state(mip.image).layout
                rcas_draws.append(rcas)
            else:
                rcas_draws.append(
                    self._prepare_copy_draw(
                        quality,
                        mip,
                        descriptor_index,
                        source_ready_in_submission=True,
                    )
                )
            screen_draws.append(self._prepare_draw(
                mip, item["target"], array_layer=int(item["array_layer"]),
                eye_index=eye_index, frame_slot=int(item["frame_slot"]),
                push_constants=item["push_constants"], clear_color=item["clear_color"],
                source_ready_in_submission=True,
            ))
        wait_semaphores = [item["wait_semaphore"] for item in draws if item["wait_semaphore"] is not None]
        submit_profile: dict[str, float] = {}
        def record_stereo(command_buffer: Any) -> None:
            for index, quality_draw in enumerate(quality_draws):
                self._record_copy_draw(command_buffer, quality_draw)
                mip_draw = rcas_draws[index]
                if use_rcas:
                    self._record_rcas_draw(command_buffer, mip_draw)
                else:
                    self._record_copy_draw(command_buffer, mip_draw)
                self._record_generate_mips(
                    command_buffer,
                    mip_draw["target"],
                    int(mip_draw["target_old_layout"]),
                )
                self._record_draw(command_buffer, screen_draws[index])

        timeline = self.context.submit_on(
            "graphics",
            record_stereo,
            wait_semaphore=wait_semaphores,
            on_submit_profile=submit_profile.update,
        )
        self.last_submit_profile = submit_profile
        if use_rcas:
            for quality, rcas, screen in zip(quality_draws, rcas_draws, screen_draws):
                self._complete_quality_mip_draw(quality, rcas, screen, timeline)
        else:
            for quality, copy_draw, screen in zip(quality_draws, rcas_draws, screen_draws):
                self._complete_draw(quality, timeline)
                self._complete_mip_draw(copy_draw, screen, timeline)
        self.quality_slot_timelines[frame_slot] = int(timeline)
        self.mip_slot_timelines[frame_slot] = int(timeline)
        self._last_submit_timeline = int(timeline)
        return int(timeline)

    def _supports_linear_blit(self, format_value: int) -> bool:
        properties = self.vk.vkGetPhysicalDeviceFormatProperties(
            self.context.physical_device, int(format_value)
        )
        required = (
            self.vk.VK_FORMAT_FEATURE_BLIT_SRC_BIT
            | self.vk.VK_FORMAT_FEATURE_BLIT_DST_BIT
            | self.vk.VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
        )
        return bool(int(properties.optimalTilingFeatures) & int(required) == int(required))

    def _prepare_copy_draw(
        self,
        source: Any,
        target: Any,
        descriptor_index: int,
        *,
        source_ready_in_submission: bool = False,
    ) -> dict[str, Any]:
        if self.copy_pipeline is None or self.copy_pipeline_layout is None:
            raise RuntimeError("Vulkan projection mip copy pass is unavailable")
        source_view = source.require_view()
        source_state = self.context.image_state(source.image)
        if (
            not source_ready_in_submission
            and source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        ):
            raise ValueError("projection mip source must be shader-readable")
        # Native mip now records copy and RCAS in the same command buffer.
        # Keep their image bindings separate so the RCAS update cannot replace
        # the copy source before that draw executes.
        descriptor_set = self.quality_descriptor_sets[descriptor_index]
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=self.sampler,
                            imageView=source_view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(target, 0)
        return {
            "source": source,
            "target": target,
            "framebuffer": framebuffer,
            "descriptor_set": descriptor_set,
            "descriptor_index": descriptor_index,
            "target_old_layout": self.context.image_state(target.image).layout,
            "source_ready_in_submission": bool(source_ready_in_submission),
        }

    def _prepare_quality_draw(
        self, source: Any, target: Any, descriptor_index: int, *, mode: str, scale: float
    ) -> dict[str, Any]:
        if self.quality_pipeline is None or self.quality_pipeline_layout is None:
            raise RuntimeError("Vulkan projection quality pass is unavailable")
        source_state = self.context.image_state(source.image)
        if source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            raise ValueError("projection quality source must be shader-readable")
        descriptor_set = self.quality_descriptor_sets[descriptor_index]
        self.vk.vkUpdateDescriptorSets(
            self.context.device, 1, [self.vk.VkWriteDescriptorSet(
                sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=0, descriptorCount=1,
                descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[self.vk.VkDescriptorImageInfo(
                    sampler=self.sampler, imageView=source.require_view(),
                    imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                )],
            )], 0, None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(target, 0)
        mode_value = 2.0 if mode == "upscale_easu" else 1.0
        payload = struct.pack(
            "<4f", 1.0 / float(source.width), 1.0 / float(source.height),
            mode_value, float(scale),
        )
        return {
            "source": source, "target": target, "framebuffer": framebuffer,
            "descriptor_set": descriptor_set, "descriptor_index": descriptor_index,
            "target_old_layout": self.context.image_state(target.image).layout,
            "pipeline": self.quality_pipeline,
            "pipeline_layout": self.quality_pipeline_layout,
            "payload": self.vk.ffi.new("char[]", payload),
        }

    def _record_copy_draw(self, command_buffer: Any, draw: dict[str, Any]) -> None:
        source = draw["source"]
        target = draw["target"]
        if draw.get("source_ready_in_submission", False):
            source_transition = self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                newLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=source.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, None, 0, None, 1, [source_transition],
            )
        old_layout = int(draw["target_old_layout"])
        transition = self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=(
                self.vk.VK_ACCESS_SHADER_READ_BIT
                if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL else 0
            ),
            dstAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            oldLayout=old_layout,
            newLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=target.image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
            ),
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            else self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 1, [transition],
        )
        self.vk.vkCmdBeginRenderPass(
            command_buffer,
            self.vk.VkRenderPassBeginInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=draw.get("render_pass", self.render_pass),
                framebuffer=draw["framebuffer"],
                renderArea=self.vk.VkRect2D(
                    offset=self.vk.VkOffset2D(x=0, y=0),
                    extent=self.vk.VkExtent2D(width=int(target.width), height=int(target.height)),
                ),
                clearValueCount=1,
                pClearValues=[self.vk.VkClearValue(
                    color=self.vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0])
                )],
            ),
            self.vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        self.vk.vkCmdSetViewport(command_buffer, 0, 1, [
            self.vk.VkViewport(x=0.0, y=0.0, width=float(target.width), height=float(target.height), minDepth=0.0, maxDepth=1.0)
        ])
        self.vk.vkCmdSetScissor(command_buffer, 0, 1, [
            self.vk.VkRect2D(offset=self.vk.VkOffset2D(x=0, y=0), extent=self.vk.VkExtent2D(width=int(target.width), height=int(target.height)))
        ])
        pipeline = draw.get("pipeline", self.copy_pipeline)
        pipeline_layout = draw.get("pipeline_layout", self.copy_pipeline_layout)
        self.vk.vkCmdBindPipeline(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        self.vk.vkCmdBindDescriptorSets(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, [draw["descriptor_set"]], 0, None)
        if "payload" in draw:
            self.vk.vkCmdPushConstants(
                command_buffer, pipeline_layout, self.vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                0, 16, draw["payload"],
            )
        self.vk.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        self.vk.vkCmdEndRenderPass(command_buffer)

    def _record_generate_mips(
        self, command_buffer: Any, image: Any, previous_layout: int
    ) -> None:
        if int(getattr(image, "mip_levels", 1)) <= 1:
            shader_read = self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                newLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=image.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, None, 0, None, 1, [shader_read],
            )
            return
        barriers = [self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=self.vk.VK_ACCESS_TRANSFER_READ_BIT,
            oldLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            newLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=image.image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
            ),
        )]
        self.vk.vkCmdPipelineBarrier(command_buffer, self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1, barriers)
        for level in range(1, int(image.mip_levels)):
            width = max(1, int(image.width) >> level)
            height = max(1, int(image.height) >> level)
            dst_transition = self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=(
                    self.vk.VK_ACCESS_SHADER_READ_BIT
                    if previous_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    else 0
                ),
                dstAccessMask=self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=previous_layout,
                newLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=image.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=level, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                if previous_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                else self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, None, 0, None, 1, [dst_transition],
            )
            self.vk.vkCmdBlitImage(
                command_buffer, image.image, self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image.image, self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                [self.vk.VkImageBlit(
                    srcSubresource=self.vk.VkImageSubresourceLayers(aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=level - 1, baseArrayLayer=0, layerCount=1),
                    srcOffsets=[self.vk.VkOffset3D(x=0, y=0, z=0), self.vk.VkOffset3D(x=max(1, int(image.width) >> (level - 1)), y=max(1, int(image.height) >> (level - 1)), z=1)],
                    dstSubresource=self.vk.VkImageSubresourceLayers(aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=level, baseArrayLayer=0, layerCount=1),
                    dstOffsets=[self.vk.VkOffset3D(x=0, y=0, z=0), self.vk.VkOffset3D(x=width, y=height, z=1)],
                )], self.vk.VK_FILTER_LINEAR,
            )
            if level + 1 < int(image.mip_levels):
                src_transition = self.vk.VkImageMemoryBarrier(
                    sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=self.vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=self.vk.VK_ACCESS_TRANSFER_READ_BIT,
                    oldLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image.image,
                    subresourceRange=self.vk.VkImageSubresourceRange(aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel=level, levelCount=1, baseArrayLayer=0, layerCount=1),
                )
                self.vk.vkCmdPipelineBarrier(command_buffer, self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT, self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1, [src_transition])
        shader_barriers = []
        for level in range(int(image.mip_levels)):
            is_last_level = level + 1 == int(image.mip_levels)
            shader_barriers.append(self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=(
                    self.vk.VK_ACCESS_TRANSFER_WRITE_BIT
                    if is_last_level else self.vk.VK_ACCESS_TRANSFER_READ_BIT
                ),
                dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=(
                    self.vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                    if is_last_level else self.vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                ),
                newLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=image.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=level, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            ))
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, None, 0, None, len(shader_barriers), shader_barriers,
        )

    def _prepare_draw(
        self,
        source: Any,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        push_constants: bytes,
        clear_color: tuple[float, float, float, float],
        source_ready_in_submission: bool = False,
        descriptor_set: Any | None = None,
        overlay: bool = False,
    ) -> dict[str, Any]:
        if (
            self.pipeline is None
            or len(push_constants) != self._PUSH_CONSTANT_SIZE
        ):
            raise RuntimeError("Vulkan projection screen pass is unavailable")
        if int(target.format) != self.target_format:
            raise ValueError("projection target format changed after pipeline creation")
        source_view = source.require_view()
        source_state = self.context.image_state(source.image)
        if (
            not source_ready_in_submission
            and source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        ):
            raise ValueError("projection screen source must be shader-readable")
        descriptor_index = (int(eye_index) * 3 + int(frame_slot)) % self._DESCRIPTOR_COUNT
        last_use = self.descriptor_timelines[descriptor_index]
        if last_use:
            self.context.wait_for_timeline(last_use)
        descriptor_set = descriptor_set or self.descriptor_sets[descriptor_index]
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=self.sampler,
                            imageView=source_view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(
            target, array_layer, overlay=overlay
        )
        return {
            "target": target,
            "array_layer": int(array_layer),
            "framebuffer": framebuffer,
            "descriptor_set": descriptor_set,
            "payload": self.vk.ffi.new("char[]", push_constants),
            "clear_color": clear_color,
            "descriptor_index": descriptor_index,
            "render_pass": self.overlay_render_pass if overlay else self.render_pass,
        }

    def _record_draw(self, command_buffer: Any, draw: dict[str, Any]) -> None:
        target = draw["target"]
        old_layout = int(draw.get("target_old_layout", self.vk.VK_IMAGE_LAYOUT_UNDEFINED))
        old_access = (
            self.vk.VK_ACCESS_SHADER_READ_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            else self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            else 0
        )
        old_stage = (
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            else self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            if old_layout == self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            else self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        )
        transition = self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=old_access,
            dstAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            oldLayout=old_layout,
            newLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=target.image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=int(draw["array_layer"]),
                layerCount=1,
            ),
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            old_stage,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 1, [transition],
        )
        self.vk.vkCmdBeginRenderPass(
            command_buffer,
            self.vk.VkRenderPassBeginInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.render_pass,
                framebuffer=draw["framebuffer"],
                renderArea=self.vk.VkRect2D(
                    offset=self.vk.VkOffset2D(x=0, y=0),
                    extent=self.vk.VkExtent2D(
                        width=int(target.width), height=int(target.height)
                    ),
                ),
                clearValueCount=1,
                pClearValues=[
                    self.vk.VkClearValue(
                        color=self.vk.VkClearColorValue(
                            float32=[float(value) for value in draw["clear_color"]]
                        )
                    )
                ],
            ),
            self.vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        self.vk.vkCmdSetViewport(
            command_buffer,
            0,
            1,
            [
                self.vk.VkViewport(
                    x=0.0,
                    y=0.0,
                    width=float(target.width),
                    height=float(target.height),
                    minDepth=0.0,
                    maxDepth=1.0,
                )
            ],
        )
        self.vk.vkCmdSetScissor(
            command_buffer,
            0,
            1,
            [
                self.vk.VkRect2D(
                    offset=self.vk.VkOffset2D(x=0, y=0),
                    extent=self.vk.VkExtent2D(
                        width=int(target.width), height=int(target.height)
                    ),
                )
            ],
        )
        self.vk.vkCmdBindPipeline(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline,
        )
        self.vk.vkCmdBindDescriptorSets(
            command_buffer,
            self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout,
            0,
            1,
            [draw["descriptor_set"]],
            0,
            None,
        )
        self.vk.vkCmdPushConstants(
            command_buffer,
            self.pipeline_layout,
            (
                self.vk.VK_SHADER_STAGE_VERTEX_BIT
                | self.vk.VK_SHADER_STAGE_FRAGMENT_BIT
            ),
            0,
            self._PUSH_CONSTANT_SIZE,
            draw["payload"],
        )
        self.vk.vkCmdDraw(command_buffer, self._VERTEX_COUNT, 1, 0, 0)
        self.vk.vkCmdEndRenderPass(command_buffer)

    def _prepare_rcas_draw(
        self,
        source: Any,
        target: Any,
        *,
        array_layer: int,
        eye_index: int,
        frame_slot: int,
        sharpness: float,
    ) -> dict[str, Any]:
        if self.rcas_pipeline is None or self.rcas_pipeline_layout is None:
            raise RuntimeError("Vulkan projection RCAS pass is unavailable")
        descriptor_index = (int(eye_index) * 3 + int(frame_slot)) % self._DESCRIPTOR_COUNT
        descriptor_set = self.rcas_descriptor_sets[descriptor_index]
        source_view = source.require_view()
        self.vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                self.vk.VkWriteDescriptorSet(
                    sType=self.vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=self.vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[
                        self.vk.VkDescriptorImageInfo(
                            sampler=self.sampler,
                            imageView=source_view,
                            imageLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        _view, framebuffer = self._target_view_and_framebuffer(target, array_layer)
        payload = struct.pack(
            "<4f",
            1.0 / float(source.width),
            1.0 / float(source.height),
            float(sharpness),
            0.0,
        )
        return {
            "source": source,
            "target": target,
            "array_layer": int(array_layer),
            "framebuffer": framebuffer,
            "descriptor_set": descriptor_set,
            "payload": self.vk.ffi.new("char[]", payload),
            "descriptor_index": descriptor_index,
        }

    def _record_rcas_draw(self, command_buffer: Any, draw: dict[str, Any]) -> None:
        source = draw["source"]
        target = draw["target"]
        source_transition = self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=self.vk.VK_ACCESS_SHADER_READ_BIT,
            oldLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            newLayout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=source.image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
            ),
        )
        target_old_layout = int(draw.get("target_old_layout", self.vk.VK_IMAGE_LAYOUT_UNDEFINED))
        target_transition = self.vk.VkImageMemoryBarrier(
            sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=(
                self.vk.VK_ACCESS_SHADER_READ_BIT
                if target_old_layout == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL else 0
            ),
            dstAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            oldLayout=target_old_layout,
            newLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
            image=target.image,
            subresourceRange=self.vk.VkImageSubresourceRange(
                aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=int(draw["array_layer"]), layerCount=1,
            ),
        )
        self.vk.vkCmdPipelineBarrier(
            command_buffer,
            self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            | self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, None, 0, None, 2, [source_transition, target_transition],
        )
        self.vk.vkCmdBeginRenderPass(
            command_buffer,
            self.vk.VkRenderPassBeginInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self.render_pass,
                framebuffer=draw["framebuffer"],
                renderArea=self.vk.VkRect2D(
                    offset=self.vk.VkOffset2D(x=0, y=0),
                    extent=self.vk.VkExtent2D(width=int(target.width), height=int(target.height)),
                ),
                clearValueCount=1,
                pClearValues=[
                    self.vk.VkClearValue(
                        color=self.vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0])
                    )
                ],
            ),
            self.vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        self.vk.vkCmdSetViewport(command_buffer, 0, 1, [
            self.vk.VkViewport(x=0.0, y=0.0, width=float(target.width), height=float(target.height), minDepth=0.0, maxDepth=1.0)
        ])
        self.vk.vkCmdSetScissor(command_buffer, 0, 1, [
            self.vk.VkRect2D(offset=self.vk.VkOffset2D(x=0, y=0), extent=self.vk.VkExtent2D(width=int(target.width), height=int(target.height)))
        ])
        self.vk.vkCmdBindPipeline(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.rcas_pipeline)
        self.vk.vkCmdBindDescriptorSets(command_buffer, self.vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.rcas_pipeline_layout, 0, 1, [draw["descriptor_set"]], 0, None)
        self.vk.vkCmdPushConstants(command_buffer, self.rcas_pipeline_layout, self.vk.VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, draw["payload"])
        self.vk.vkCmdDraw(command_buffer, 3, 1, 0, 0)
        self.vk.vkCmdEndRenderPass(command_buffer)

    def _complete_draw(self, draw: dict[str, Any], timeline: int) -> None:
        target = draw["target"]
        self.context.register_image_state(
            target.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.descriptor_timelines[draw["descriptor_index"]] = int(timeline)

    def _complete_rcas_draw(self, draw: dict[str, Any], timeline: int) -> None:
        source = draw["source"]
        target = draw["target"]
        self.context.register_image_state(
            source.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.context.register_image_state(
            target.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.rcas_descriptor_timelines[draw["descriptor_index"]] = int(timeline)

    def _complete_mip_draw(
        self, copy_draw: dict[str, Any], screen_draw: dict[str, Any], timeline: int
    ) -> None:
        mip = copy_draw["target"]
        self.context.register_image_state(
            mip.image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self._complete_draw(screen_draw, timeline)
        self.quality_descriptor_timelines[copy_draw["descriptor_index"]] = int(timeline)

    def _complete_quality_mip_draw(
        self,
        quality_draw: dict[str, Any],
        rcas_draw: dict[str, Any],
        screen_draw: dict[str, Any],
        timeline: int,
    ) -> None:
        self.context.register_image_state(
            quality_draw["target"].image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self.context.register_image_state(
            rcas_draw["target"].image,
            ImageState(
                layout=self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=self.vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=self.vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        self._complete_draw(screen_draw, timeline)
        self.quality_descriptor_timelines[quality_draw["descriptor_index"]] = int(timeline)
        self.rcas_descriptor_timelines[rcas_draw["descriptor_index"]] = int(timeline)

    def close(self) -> None:
        if self.context.device is not None:
            for images in self.quality_images.values():
                for image in images:
                    image.close()
            for images in self.mip_images.values():
                for image in images:
                    image.close()
            for framebuffer in self.framebuffers.values():
                self.vk.vkDestroyFramebuffer(self.context.device, framebuffer, None)
            for framebuffer in self.overlay_framebuffers.values():
                self.vk.vkDestroyFramebuffer(self.context.device, framebuffer, None)
            for view in self.image_views.values():
                self.vk.vkDestroyImageView(self.context.device, view, None)
            if self.pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.pipeline, None)
            if self.rcas_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.rcas_pipeline, None)
            if self.copy_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.copy_pipeline, None)
            if self.quality_pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.quality_pipeline, None)
            if self.pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.pipeline_layout, None
                )
            if self.rcas_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.rcas_pipeline_layout, None
                )
            if self.copy_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.copy_pipeline_layout, None
                )
            if self.quality_pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.quality_pipeline_layout, None
                )
            if self.render_pass is not None:
                self.vk.vkDestroyRenderPass(self.context.device, self.render_pass, None)
            if self.overlay_render_pass is not None:
                self.vk.vkDestroyRenderPass(
                    self.context.device, self.overlay_render_pass, None
                )
            if self.sampler is not None:
                self.vk.vkDestroySampler(self.context.device, self.sampler, None)
            for sampler, _timeline in self._retired_samplers:
                self.vk.vkDestroySampler(self.context.device, sampler, None)
            if self.descriptor_pool is not None:
                self.vk.vkDestroyDescriptorPool(
                    self.context.device, self.descriptor_pool, None
                )
            if self.descriptor_set_layout is not None:
                self.vk.vkDestroyDescriptorSetLayout(
                    self.context.device, self.descriptor_set_layout, None
                )
            for module in self.shader_modules:
                self.vk.vkDestroyShaderModule(self.context.device, module, None)
        self.framebuffers.clear()
        self.overlay_framebuffers.clear()
        self.image_views.clear()
        self.quality_images.clear()
        self.mip_images.clear()
        self.shader_modules.clear()
        self.descriptor_sets.clear()
        self.rcas_descriptor_sets.clear()
        self.quality_descriptor_sets.clear()
        self.pipeline = None
        self.pipeline_layout = None
        self.rcas_pipeline = None
        self.rcas_pipeline_layout = None
        self.copy_pipeline = None
        self.copy_pipeline_layout = None
        self.quality_pipeline = None
        self.quality_pipeline_layout = None
        self.render_pass = None
        self.overlay_render_pass = None
        self.sampler = None
        self._retired_samplers.clear()
        self.descriptor_pool = None
        self.descriptor_set_layout = None
