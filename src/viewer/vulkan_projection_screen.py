from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

from viewer.vulkan_compute_pipeline import read_spirv_words
from viewer.vulkan_context import ImageState
from viewer.vulkan_descriptors import DescriptorBinding, create_descriptor_set_layout


class VulkanProjectionScreenPass:
    """Rasterize the world-space screen directly into one projection layer."""

    _SEGMENTS = 48
    _VERTEX_COUNT = (_SEGMENTS + 1) * 2
    _PUSH_CONSTANT_SIZE = 128
    _DESCRIPTOR_COUNT = 6

    def __init__(self, context: Any, target_format: int) -> None:
        self.context = context
        self.vk = context.vk
        self.target_format = int(target_format)
        self.shader_modules: list[Any] = []
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_sets: list[Any] = []
        self.descriptor_timelines = [0] * self._DESCRIPTOR_COUNT
        self.sampler = None
        self.render_pass = None
        self.pipeline_layout = None
        self.pipeline = None
        self.image_views: dict[tuple[int, int], Any] = {}
        self.framebuffers: dict[tuple[int, int, int, int], Any] = {}
        try:
            self._create()
        except Exception:
            self.close()
            raise

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
                maxSets=self._DESCRIPTOR_COUNT,
                poolSizeCount=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        descriptorCount=self._DESCRIPTOR_COUNT,
                    )
                ],
            ),
            None,
        )
        self.descriptor_sets = list(
            vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.descriptor_pool,
                    descriptorSetCount=self._DESCRIPTOR_COUNT,
                    pSetLayouts=[self.descriptor_set_layout] * self._DESCRIPTOR_COUNT,
                ),
            )
        )
        self.sampler = vk.vkCreateSampler(
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
                minLod=0.0,
                maxLod=0.0,
            ),
            None,
        )
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
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                        offset=0,
                        size=self._PUSH_CONSTANT_SIZE,
                    )
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
                    layout=self.pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]

    def _target_view_and_framebuffer(
        self, target: Any, array_layer: int
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
        framebuffer = self.framebuffers.get(framebuffer_key)
        if framebuffer is None:
            framebuffer = vk.vkCreateFramebuffer(
                self.context.device,
                vk.VkFramebufferCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    renderPass=self.render_pass,
                    attachmentCount=1,
                    pAttachments=[view],
                    width=int(target.width),
                    height=int(target.height),
                    layers=1,
                ),
                None,
            )
            self.framebuffers[framebuffer_key] = framebuffer
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
        if self.pipeline is None or len(push_constants) != self._PUSH_CONSTANT_SIZE:
            raise RuntimeError("Vulkan projection screen pass is unavailable")
        if int(target.format) != self.target_format:
            raise ValueError("projection target format changed after pipeline creation")
        source_view = source.require_view()
        source_state = self.context.image_state(source.image)
        if source_state.layout != self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            raise ValueError("projection screen source must be shader-readable")
        descriptor_index = (int(eye_index) * 3 + int(frame_slot)) % self._DESCRIPTOR_COUNT
        last_use = self.descriptor_timelines[descriptor_index]
        if last_use:
            self.context.wait_for_timeline(last_use)
        descriptor_set = self.descriptor_sets[descriptor_index]
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
        payload = self.vk.ffi.new("char[]", push_constants)

        def record(command_buffer: Any) -> None:
            transition = self.vk.VkImageMemoryBarrier(
                sType=self.vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=self.vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                oldLayout=self.vk.VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout=self.vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                srcQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=self.vk.VK_QUEUE_FAMILY_IGNORED,
                image=target.image,
                subresourceRange=self.vk.VkImageSubresourceRange(
                    aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0,
                    levelCount=1,
                    baseArrayLayer=int(array_layer),
                    layerCount=1,
                ),
            )
            self.vk.vkCmdPipelineBarrier(
                command_buffer,
                self.vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                self.vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, 0, None, 0, None, 1, [transition],
            )
            self.vk.vkCmdBeginRenderPass(
                command_buffer,
                self.vk.VkRenderPassBeginInfo(
                    sType=self.vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    renderPass=self.render_pass,
                    framebuffer=framebuffer,
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
                                float32=[float(value) for value in clear_color]
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
                [descriptor_set],
                0,
                None,
            )
            self.vk.vkCmdPushConstants(
                command_buffer,
                self.pipeline_layout,
                self.vk.VK_SHADER_STAGE_VERTEX_BIT,
                0,
                self._PUSH_CONSTANT_SIZE,
                payload,
            )
            self.vk.vkCmdDraw(command_buffer, self._VERTEX_COUNT, 1, 0, 0)
            self.vk.vkCmdEndRenderPass(command_buffer)

        timeline = self.context.submit_on(
            "graphics", record, wait_semaphore=wait_semaphore
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
        self.descriptor_timelines[descriptor_index] = int(timeline)
        return int(timeline)

    def close(self) -> None:
        if self.context.device is not None:
            for framebuffer in self.framebuffers.values():
                self.vk.vkDestroyFramebuffer(self.context.device, framebuffer, None)
            for view in self.image_views.values():
                self.vk.vkDestroyImageView(self.context.device, view, None)
            if self.pipeline is not None:
                self.vk.vkDestroyPipeline(self.context.device, self.pipeline, None)
            if self.pipeline_layout is not None:
                self.vk.vkDestroyPipelineLayout(
                    self.context.device, self.pipeline_layout, None
                )
            if self.render_pass is not None:
                self.vk.vkDestroyRenderPass(self.context.device, self.render_pass, None)
            if self.sampler is not None:
                self.vk.vkDestroySampler(self.context.device, self.sampler, None)
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
        self.image_views.clear()
        self.shader_modules.clear()
        self.descriptor_sets.clear()
        self.pipeline = None
        self.pipeline_layout = None
        self.render_pass = None
        self.sampler = None
        self.descriptor_pool = None
        self.descriptor_set_layout = None
