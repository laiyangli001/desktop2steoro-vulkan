from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from viewer.vulkan_compute_pipeline import read_spirv_words
from viewer.vulkan_context import ImageState
from viewer.vulkan_descriptors import (
    DescriptorBinding,
    VulkanStorageBuffer,
    create_descriptor_set_layout,
)


class VulkanMultiviewEyeDiagnosticPass:
    """Draw red/green views into one two-layer target using gl_ViewIndex."""

    def __init__(self, context: Any, target_format: int) -> None:
        self.context = context
        self.vk = context.vk
        self.target_format = int(target_format)
        self.shader_modules: list[Any] = []
        self.render_pass = None
        self.pipeline_layout = None
        self.pipeline = None
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_set = None
        self.view_counts: VulkanStorageBuffer | None = None
        self.image_views: dict[int, Any] = {}
        self.framebuffers: dict[tuple[int, int, int], Any] = {}
        try:
            self._create()
        except Exception:
            self.close()
            raise

    def _shader_module(self, path: Path) -> Any:
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
        vertex = self._shader_module(shader_root / "d2s_multiview_eye_diag_vert.spv")
        fragment = self._shader_module(shader_root / "d2s_multiview_eye_diag_frag.spv")
        attachment = vk.VkAttachmentDescription(
            format=self.target_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        reference = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[reference],
        )
        multiview = vk.VkRenderPassMultiviewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
            subpassCount=1,
            pViewMasks=[0x3],
            correlationMaskCount=1,
            pCorrelationMasks=[0x3],
        )
        self.render_pass = vk.vkCreateRenderPass(
            self.context.device,
            vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                pNext=multiview,
                attachmentCount=1,
                pAttachments=[attachment],
                subpassCount=1,
                pSubpasses=[subpass],
            ),
            None,
        )
        self.descriptor_set_layout = create_descriptor_set_layout(
            self.context,
            [
                DescriptorBinding(
                    0,
                    vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    stage_flags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                )
            ],
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.context.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                    )
                ],
            ),
            None,
        )
        self.descriptor_set = list(
            vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.descriptor_pool,
                    descriptorSetCount=1,
                    pSetLayouts=[self.descriptor_set_layout],
                ),
            )
        )[0]
        self.view_counts = VulkanStorageBuffer(self.context, 8)
        vk.vkUpdateDescriptorSets(
            self.context.device,
            1,
            [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=self.descriptor_set,
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[
                        vk.VkDescriptorBufferInfo(
                            buffer=self.view_counts.buffer,
                            offset=0,
                            range=8,
                        )
                    ],
                )
            ],
            0,
            None,
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.context.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_set_layout],
            ),
            None,
        )
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=vertex,
                pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=fragment,
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
                    stageCount=2,
                    pStages=stages,
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
                    layout=self.pipeline_layout,
                    renderPass=self.render_pass,
                    subpass=0,
                    basePipelineIndex=-1,
                )
            ],
            None,
        )[0]

    def _framebuffer(self, target: Any) -> Any:
        vk = self.vk
        image_key = id(target.image)
        view = self.image_views.get(image_key)
        if view is None:
            view = vk.vkCreateImageView(
                self.context.device,
                vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=target.image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                    format=self.target_format,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=2,
                    ),
                ),
                None,
            )
            self.image_views[image_key] = view
        key = (image_key, int(target.width), int(target.height))
        framebuffer = self.framebuffers.get(key)
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
            self.framebuffers[key] = framebuffer
        return framebuffer

    def submit(self, target: Any) -> int:
        vk = self.vk
        framebuffer = self._framebuffer(target)
        self.view_counts.write_bytes(b"\0" * 8)

        def record(command_buffer: Any) -> None:
            vk.vkCmdBeginRenderPass(
                command_buffer,
                vk.VkRenderPassBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    renderPass=self.render_pass,
                    framebuffer=framebuffer,
                    renderArea=vk.VkRect2D(
                        offset=vk.VkOffset2D(x=0, y=0),
                        extent=vk.VkExtent2D(
                            width=int(target.width), height=int(target.height)
                        ),
                    ),
                    clearValueCount=1,
                    pClearValues=[
                        vk.VkClearValue(color=vk.VkClearColorValue(float32=[0, 0, 0, 1]))
                    ],
                ),
                vk.VK_SUBPASS_CONTENTS_INLINE,
            )
            vk.vkCmdSetViewport(
                command_buffer,
                0,
                1,
                [
                    vk.VkViewport(
                        x=0.0,
                        y=0.0,
                        width=float(target.width),
                        height=float(target.height),
                        minDepth=0.0,
                        maxDepth=1.0,
                    )
                ],
            )
            vk.vkCmdSetScissor(
                command_buffer,
                0,
                1,
                [
                    vk.VkRect2D(
                        offset=vk.VkOffset2D(x=0, y=0),
                        extent=vk.VkExtent2D(
                            width=int(target.width), height=int(target.height)
                        ),
                    )
                ],
            )
            vk.vkCmdBindPipeline(
                command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline
            )
            vk.vkCmdBindDescriptorSets(
                command_buffer,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipeline_layout,
                0,
                1,
                [self.descriptor_set],
                0,
                None,
            )
            vk.vkCmdDraw(command_buffer, 3, 1, 0, 0)
            vk.vkCmdEndRenderPass(command_buffer)
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_HOST_BIT,
                0,
                0,
                None,
                1,
                [
                    vk.VkBufferMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        buffer=self.view_counts.buffer,
                        offset=0,
                        size=8,
                    )
                ],
                0,
                None,
            )

        timeline = self.context.submit_on("graphics", record)
        self.context.register_image_state(
            target.image,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.context.queue_family_index,
            ),
        )
        return int(timeline)

    def read_view_counts(self) -> tuple[int, int]:
        payload = self.view_counts.read_bytes(8)
        return struct.unpack("<2I", payload)

    def close(self) -> None:
        if self.context.device is None:
            return
        vk = self.vk
        for framebuffer in self.framebuffers.values():
            vk.vkDestroyFramebuffer(self.context.device, framebuffer, None)
        self.framebuffers.clear()
        for view in self.image_views.values():
            vk.vkDestroyImageView(self.context.device, view, None)
        self.image_views.clear()
        if self.pipeline is not None:
            vk.vkDestroyPipeline(self.context.device, self.pipeline, None)
            self.pipeline = None
        if self.pipeline_layout is not None:
            vk.vkDestroyPipelineLayout(self.context.device, self.pipeline_layout, None)
            self.pipeline_layout = None
        if self.view_counts is not None:
            self.view_counts.close()
            self.view_counts = None
        if self.descriptor_pool is not None:
            vk.vkDestroyDescriptorPool(self.context.device, self.descriptor_pool, None)
            self.descriptor_pool = None
        if self.descriptor_set_layout is not None:
            vk.vkDestroyDescriptorSetLayout(
                self.context.device, self.descriptor_set_layout, None
            )
            self.descriptor_set_layout = None
        if self.render_pass is not None:
            vk.vkDestroyRenderPass(self.context.device, self.render_pass, None)
            self.render_pass = None
        for module in reversed(self.shader_modules):
            vk.vkDestroyShaderModule(self.context.device, module, None)
        self.shader_modules.clear()
