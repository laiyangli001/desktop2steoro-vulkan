#include "vulkan_ffmpeg_bridge.h"

#if defined(_WIN32) && !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
struct Encoder {
    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    AVBufferRef* rgba_frames = nullptr;
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* acquired = nullptr;
    AVFrame* rgba_acquired = nullptr;
    int width = 0;
    int height = 0;
    int fps = 0;
    VkQueue prepare_queue = VK_NULL_HANDLE;
    uint32_t prepare_queue_family = VK_QUEUE_FAMILY_IGNORED;
    VkQueue compute_queue = VK_NULL_HANDLE;
    uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
    VkCommandPool prepare_pool = VK_NULL_HANDLE;
    VkCommandPool compute_pool = VK_NULL_HANDLE;
    std::unordered_map<AVVkFrame*, VkCommandBuffer> prepare_buffers;

    VkPipeline convert_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout convert_pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout convert_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool convert_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet convert_descriptor_set = VK_NULL_HANDLE;
    VkShaderModule convert_shader = VK_NULL_HANDLE;
    VkImage convert_y_image = VK_NULL_HANDLE;
    VkDeviceMemory convert_y_memory = VK_NULL_HANDLE;
    VkImageView convert_y_view = VK_NULL_HANDLE;
    VkImage convert_uv_image = VK_NULL_HANDLE;
    VkDeviceMemory convert_uv_memory = VK_NULL_HANDLE;
    VkImageView convert_uv_view = VK_NULL_HANDLE;
    VkImageLayout convert_y_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout convert_uv_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::unordered_map<VkImage, VkImageView> rgba_views;
    std::unordered_map<VkImage, VkCommandBuffer> convert_buffers;
    std::unordered_map<VkImage, VkFence> convert_fences;
    std::unordered_map<VkImage, VkCommandBuffer> encode_acquire_buffers;
    std::unordered_map<VkImage, VkFence> encode_acquire_fences;
};

constexpr unsigned int D2S_EXTERNAL_HANDLE_WIN32 = 1;
constexpr unsigned int D2S_EXTERNAL_HANDLE_FD = 2;

void write_message(char* output, int capacity, const char* message) {
    if (!output || capacity <= 0) return;
    std::snprintf(output, static_cast<std::size_t>(capacity), "%s", message);
}

bool find_memory_type(Encoder* encoder, uint32_t bits, VkMemoryPropertyFlags properties,
                       uint32_t* result) {
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(vulkan->phys_dev, &memory_properties);
    for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
        if ((bits & (1u << index)) &&
            (memory_properties.memoryTypes[index].propertyFlags & properties) == properties) {
            *result = index;
            return true;
        }
    }
    return false;
}

bool create_storage_image(Encoder* encoder, VkFormat format, uint32_t width,
                          uint32_t height, VkImage* image, VkDeviceMemory* memory) {
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(vulkan->act_dev, &image_info, vulkan->alloc, image) != VK_SUCCESS)
        return false;
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(vulkan->act_dev, *image, &requirements);
    uint32_t memory_type = 0;
    if (!find_memory_type(encoder, requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &memory_type))
        return false;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(vulkan->act_dev, &allocate, vulkan->alloc, memory) != VK_SUCCESS)
        return false;
    return vkBindImageMemory(vulkan->act_dev, *image, *memory, 0) == VK_SUCCESS;
}

VkImageView create_color_view(Encoder* encoder, VkImage image, VkFormat format) {
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    return vkCreateImageView(vulkan->act_dev, &view_info, vulkan->alloc, &view) == VK_SUCCESS
        ? view : VK_NULL_HANDLE;
}

bool load_convert_shader(std::vector<uint32_t>* code) {
    std::vector<std::filesystem::path> candidates;
    if (const char* configured = std::getenv("D2S_RGB_TO_NV12_SPV"))
        candidates.emplace_back(configured);
    const auto cwd = std::filesystem::current_path();
    candidates.emplace_back(cwd / "src/desktop2steoro/shaders/d2s_rgb_to_nv12.spv");
    candidates.emplace_back(cwd / "desktop2steoro-vulkan/src/desktop2steoro/shaders/d2s_rgb_to_nv12.spv");
    for (const auto& path : candidates) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) continue;
        const auto size = file.tellg();
        if (size <= 0 || (size % static_cast<std::streamoff>(sizeof(uint32_t))) != 0)
            continue;
        code->resize(static_cast<size_t>(size) / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code->data()), size);
        if (file) return true;
    }
    return false;
}

void destroy_convert_resources(Encoder* encoder) {
    if (!encoder || !encoder->device) return;
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    const VkDevice device = vulkan->act_dev;
    for (auto& item : encoder->rgba_views)
        vkDestroyImageView(device, item.second, vulkan->alloc);
    encoder->rgba_views.clear();
    for (auto& item : encoder->convert_fences)
        vkDestroyFence(device, item.second, vulkan->alloc);
    for (auto& item : encoder->encode_acquire_fences)
        vkDestroyFence(device, item.second, vulkan->alloc);
    encoder->convert_fences.clear();
    encoder->convert_buffers.clear();
    encoder->encode_acquire_fences.clear();
    encoder->encode_acquire_buffers.clear();
    if (encoder->convert_descriptor_pool)
        vkDestroyDescriptorPool(device, encoder->convert_descriptor_pool, vulkan->alloc);
    if (encoder->convert_pipeline)
        vkDestroyPipeline(device, encoder->convert_pipeline, vulkan->alloc);
    if (encoder->convert_pipeline_layout)
        vkDestroyPipelineLayout(device, encoder->convert_pipeline_layout, vulkan->alloc);
    if (encoder->convert_descriptor_layout)
        vkDestroyDescriptorSetLayout(device, encoder->convert_descriptor_layout, vulkan->alloc);
    if (encoder->convert_shader)
        vkDestroyShaderModule(device, encoder->convert_shader, vulkan->alloc);
    if (encoder->convert_y_view)
        vkDestroyImageView(device, encoder->convert_y_view, vulkan->alloc);
    if (encoder->convert_uv_view)
        vkDestroyImageView(device, encoder->convert_uv_view, vulkan->alloc);
    if (encoder->convert_y_image)
        vkDestroyImage(device, encoder->convert_y_image, vulkan->alloc);
    if (encoder->convert_uv_image)
        vkDestroyImage(device, encoder->convert_uv_image, vulkan->alloc);
    if (encoder->convert_y_memory)
        vkFreeMemory(device, encoder->convert_y_memory, vulkan->alloc);
    if (encoder->convert_uv_memory)
        vkFreeMemory(device, encoder->convert_uv_memory, vulkan->alloc);
    encoder->convert_descriptor_pool = VK_NULL_HANDLE;
    encoder->convert_pipeline = VK_NULL_HANDLE;
    encoder->convert_pipeline_layout = VK_NULL_HANDLE;
    encoder->convert_descriptor_layout = VK_NULL_HANDLE;
    encoder->convert_shader = VK_NULL_HANDLE;
    encoder->convert_y_view = VK_NULL_HANDLE;
    encoder->convert_uv_view = VK_NULL_HANDLE;
    encoder->convert_y_image = VK_NULL_HANDLE;
    encoder->convert_uv_image = VK_NULL_HANDLE;
    encoder->convert_y_memory = VK_NULL_HANDLE;
    encoder->convert_uv_memory = VK_NULL_HANDLE;
}

bool initialize_convert_pipeline(Encoder* encoder) {
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    const VkDevice device = vulkan->act_dev;
    if (!create_storage_image(encoder, VK_FORMAT_R8_UNORM, encoder->width,
                              encoder->height, &encoder->convert_y_image,
                              &encoder->convert_y_memory)) return false;
    if (!create_storage_image(encoder, VK_FORMAT_R8G8_UNORM, encoder->width / 2,
                              encoder->height / 2, &encoder->convert_uv_image,
                              &encoder->convert_uv_memory)) return false;
    encoder->convert_y_view = create_color_view(encoder, encoder->convert_y_image, VK_FORMAT_R8_UNORM);
    encoder->convert_uv_view = create_color_view(encoder, encoder->convert_uv_image, VK_FORMAT_R8G8_UNORM);
    if (!encoder->convert_y_view || !encoder->convert_uv_view) return false;
    std::vector<uint32_t> shader_code;
    if (!load_convert_shader(&shader_code)) return false;
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = shader_code.size() * sizeof(uint32_t);
    shader_info.pCode = shader_code.data();
    if (vkCreateShaderModule(device, &shader_info, vulkan->alloc, &encoder->convert_shader) != VK_SUCCESS)
        return false;
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t index = 0; index < 3; ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &layout_info, vulkan->alloc,
                                    &encoder->convert_descriptor_layout) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &encoder->convert_descriptor_layout;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, vulkan->alloc,
                               &encoder->convert_pipeline_layout) != VK_SUCCESS) return false;
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = encoder->convert_shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage;
    pipeline_info.layout = encoder->convert_pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 vulkan->alloc, &encoder->convert_pipeline) != VK_SUCCESS) return false;
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device, &pool_info, vulkan->alloc,
                                &encoder->convert_descriptor_pool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = encoder->convert_descriptor_pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &encoder->convert_descriptor_layout;
    return vkAllocateDescriptorSets(device, &allocate, &encoder->convert_descriptor_set) == VK_SUCCESS;
}

void destroy_encoder(Encoder* encoder) {
    if (!encoder) return;
    if (encoder->device) {
        auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
        auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
        vkDeviceWaitIdle(vulkan->act_dev);
        destroy_convert_resources(encoder);
        if (encoder->compute_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(vulkan->act_dev, encoder->compute_pool, vulkan->alloc);
        if (encoder->prepare_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(vulkan->act_dev, encoder->prepare_pool, vulkan->alloc);
    }
    av_frame_free(&encoder->acquired);
    av_frame_free(&encoder->rgba_acquired);
    av_packet_free(&encoder->packet);
    avcodec_free_context(&encoder->codec);
    av_buffer_unref(&encoder->frames);
    av_buffer_unref(&encoder->rgba_frames);
    av_buffer_unref(&encoder->device);
    delete encoder;
}

bool prepare_frame_for_cuda(Encoder* encoder, AVVkFrame* source) {
    if (!encoder || !source || encoder->prepare_queue == VK_NULL_HANDLE)
        return false;
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    const VkDevice device = vulkan->act_dev;
    auto submit2 = reinterpret_cast<PFN_vkQueueSubmit2>(vkGetDeviceProcAddr(device, "vkQueueSubmit2"));
    if (!submit2) return false;
    VkCommandBuffer command = VK_NULL_HANDLE;
    const auto found = encoder->prepare_buffers.find(source);
    if (found == encoder->prepare_buffers.end()) {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = encoder->prepare_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocate, &command) != VK_SUCCESS)
            return false;
        encoder->prepare_buffers.emplace(source, command);
    } else {
        command = found->second;
        if (vkResetCommandBuffer(command, 0) != VK_SUCCESS)
            return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS)
        return false;
    VkImageMemoryBarrier2 barriers[2]{};
    unsigned int count = 0;
    for (unsigned int index = 0; index < 2 && source->img[index]; ++index) {
        barriers[count] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barriers[count].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barriers[count].srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        barriers[count].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barriers[count].dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barriers[count].oldLayout = source->layout[index];
        barriers[count].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[count].srcQueueFamilyIndex = source->queue_family[index];
        barriers[count].dstQueueFamilyIndex = source->queue_family[index] == VK_QUEUE_FAMILY_IGNORED
            ? VK_QUEUE_FAMILY_IGNORED : encoder->prepare_queue_family;
        barriers[count].image = source->img[index];
        barriers[count].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[count].subresourceRange.levelCount = 1;
        barriers[count].subresourceRange.layerCount = 1;
        ++count;
    }
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = count;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command, &dependency);
    if (vkEndCommandBuffer(command) != VK_SUCCESS)
        return false;
    VkSemaphoreSubmitInfo waits[2]{};
    VkSemaphoreSubmitInfo signals[2]{};
    unsigned int waits_count = 0;
    for (unsigned int index = 0; index < count; ++index) {
        if (source->sem_value[index]) {
            waits[waits_count] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            waits[waits_count].semaphore = source->sem[index];
            waits[waits_count].value = source->sem_value[index];
            waits[waits_count].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            ++waits_count;
        }
        signals[index] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signals[index].semaphore = source->sem[index];
        signals[index].value = ++source->sem_value[index];
        signals[index].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        source->layout[index] = VK_IMAGE_LAYOUT_GENERAL;
        source->access[index] = VK_ACCESS_2_MEMORY_WRITE_BIT;
        source->queue_family[index] = encoder->prepare_queue_family;
    }
    VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_info.commandBuffer = command;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = waits_count;
    submit.pWaitSemaphoreInfos = waits;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command_info;
    submit.signalSemaphoreInfoCount = count;
    submit.pSignalSemaphoreInfos = signals;
    return submit2(encoder->prepare_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
}

bool export_vk_frame_handles(Encoder* encoder, const AVVkFrame* source,
                             D2SVulkanVideoFrame* destination) {
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan_context = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    const VkDevice device = vulkan_context->act_dev;
    if (device == VK_NULL_HANDLE) return false;
#ifdef _WIN32
    auto get_memory = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    auto get_semaphore = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
#else
    auto get_memory = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    auto get_semaphore = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
#endif
    if (!get_memory || !get_semaphore) return false;
    for (unsigned int index = 0; index < destination->plane_count; ++index) {
#ifdef _WIN32
        HANDLE memory_handle = nullptr;
        VkMemoryGetWin32HandleInfoKHR memory_info{
            VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            nullptr,
            source->mem[index],
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
        };
        if (get_memory(device, &memory_info, &memory_handle) != VK_SUCCESS || !memory_handle)
            return false;
        HANDLE semaphore_handle = nullptr;
        VkSemaphoreGetWin32HandleInfoKHR semaphore_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            nullptr,
            source->sem[index],
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
        };
        if (get_semaphore(device, &semaphore_info, &semaphore_handle) != VK_SUCCESS || !semaphore_handle) {
            CloseHandle(memory_handle);
            return false;
        }
        destination->external_memory_handle[index] =
            static_cast<long long>(reinterpret_cast<intptr_t>(memory_handle));
        destination->external_semaphore_handle[index] =
            static_cast<long long>(reinterpret_cast<intptr_t>(semaphore_handle));
#else
        int memory_handle = -1;
        VkMemoryGetFdInfoKHR memory_info{
            VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            nullptr,
            source->mem[index],
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        if (get_memory(device, &memory_info, &memory_handle) != VK_SUCCESS || memory_handle < 0)
            return false;
        int semaphore_handle = -1;
        VkSemaphoreGetFdInfoKHR semaphore_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            nullptr,
            source->sem[index],
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        if (get_semaphore(device, &semaphore_info, &semaphore_handle) != VK_SUCCESS || semaphore_handle < 0) {
            close(memory_handle);
            return false;
        }
        destination->external_memory_handle[index] = memory_handle;
        destination->external_semaphore_handle[index] = semaphore_handle;
#endif
        destination->semaphore_value[index] = source->sem_value[index];
    }
#ifdef _WIN32
    destination->external_handle_type = D2S_EXTERNAL_HANDLE_WIN32;
#else
    destination->external_handle_type = D2S_EXTERNAL_HANDLE_FD;
#endif
    return true;
}

void close_exported_handles(D2SVulkanVideoFrame* frame) {
    if (!frame) return;
    for (unsigned int index = 0; index < 2; ++index) {
#ifdef _WIN32
        if (frame->external_memory_handle[index])
            CloseHandle(reinterpret_cast<HANDLE>(static_cast<intptr_t>(frame->external_memory_handle[index])));
        if (frame->external_semaphore_handle[index])
            CloseHandle(reinterpret_cast<HANDLE>(static_cast<intptr_t>(frame->external_semaphore_handle[index])));
#else
        if (frame->external_memory_handle[index] > 0) close(static_cast<int>(frame->external_memory_handle[index]));
        if (frame->external_semaphore_handle[index] > 0) close(static_cast<int>(frame->external_semaphore_handle[index]));
#endif
        frame->external_memory_handle[index] = 0;
        frame->external_semaphore_handle[index] = 0;
    }
}

void copy_vk_frame(Encoder* encoder, const AVVkFrame* source, D2SVulkanVideoFrame* destination,
                   int width, int height, AVPixelFormat pixel_format = AV_PIX_FMT_NV12) {
    std::memset(destination, 0, sizeof(*destination));
    destination->width = static_cast<unsigned int>(width);
    destination->height = static_cast<unsigned int>(height);
    destination->plane_count = 0;
    destination->slot_id = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(source));
    for (unsigned int index = 0; index < 2; ++index) {
        if (!source->img[index]) continue;
        destination->image[index] = reinterpret_cast<void*>(source->img[index]);
        destination->memory[index] = reinterpret_cast<void*>(source->mem[index]);
        destination->memory_size[index] = static_cast<unsigned long long>(source->size[index]);
        destination->memory_offset[index] = static_cast<long long>(source->offset[index]);
        destination->layout[index] = static_cast<unsigned int>(source->layout[index]);
        destination->plane_count = index + 1;
    }
    const VkFormat* formats = av_vkfmt_from_pixfmt(pixel_format);
    if (formats) {
        for (unsigned int index = 0; index < destination->plane_count; ++index)
            destination->format[index] = static_cast<unsigned int>(formats[index]);
    }
    if (destination->plane_count > 0 && !export_vk_frame_handles(encoder, source, destination))
        close_exported_handles(destination);
}
}

extern "C" int d2s_vulkan_ffmpeg_bridge_abi_version() { return 5; }

extern "C" int d2s_vulkan_ffmpeg_bridge_probe(char* output, int capacity) {
    const AVCodec* h264 = avcodec_find_encoder_by_name("h264_vulkan");
    const AVCodec* hevc = avcodec_find_encoder_by_name("hevc_vulkan");
    if (!h264 && !hevc) {
        write_message(output, capacity, "FFmpeg Vulkan encoders are unavailable");
        return 0;
    }
    AVBufferRef* device = nullptr;
    const int result = av_hwdevice_ctx_create(
        &device, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0);
    if (result < 0) {
        char error[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(result, error, sizeof(error));
        write_message(output, capacity, error);
        return 0;
    }
    av_buffer_unref(&device);
    write_message(output, capacity,
        "FFmpeg Vulkan device and encoder discovery passed; frame-pool ABI available");
    return 1;
}

extern "C" void* d2s_vulkan_ffmpeg_encoder_create(
    void* vk_instance,
    void* vk_physical_device,
    void* vk_device,
    void* vk_queue,
    int queue_family,
    int width,
    int height,
    int fps,
    int target_bitrate,
    int peak_bitrate,
    int hevc) {
    if (width < 1 || height < 1 || fps < 1)
        return nullptr;

    const char* encoder_name = hevc ? "hevc_vulkan" : "h264_vulkan";
    const AVCodec* codec = avcodec_find_encoder_by_name(encoder_name);
    if (!codec) return nullptr;

    auto* result = new Encoder();
    result->width = width;
    result->height = height;
    result->fps = fps;
    const bool use_external_device = vk_instance && vk_physical_device &&
        vk_device && vk_queue;
    if (use_external_device) {
        // The caller must have created this VkDevice with the required Vulkan
        // Video extensions. Most viewer contexts do not, so normal streaming
        // uses the FFmpeg-owned branch below and shares frame memory explicitly.
        result->device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
        if (!result->device) {
            destroy_encoder(result);
            return nullptr;
        }
        auto* device_context = reinterpret_cast<AVHWDeviceContext*>(result->device->data);
        auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
        vulkan->get_proc_addr = vkGetInstanceProcAddr;
        vulkan->inst = reinterpret_cast<VkInstance>(vk_instance);
        vulkan->phys_dev = reinterpret_cast<VkPhysicalDevice>(vk_physical_device);
        vulkan->act_dev = reinterpret_cast<VkDevice>(vk_device);
        vulkan->device_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        vulkan->qf[0].idx = queue_family;
        vulkan->qf[0].num = 1;
        vulkan->qf[0].flags = static_cast<VkQueueFlagBits>(
            VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_VIDEO_ENCODE_BIT_KHR);
        vulkan->qf[0].video_caps = hevc
            ? VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR
            : VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
        vulkan->nb_qf = 1;
        if (av_hwdevice_ctx_init(result->device) < 0) {
            destroy_encoder(result);
            return nullptr;
        }
    } else if (av_hwdevice_ctx_create(
                   &result->device, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0) < 0) {
        // FFmpeg creates a device with the extensions its Vulkan backend needs.
        // This is the safe default for a standalone encoder bridge.
        destroy_encoder(result);
        return nullptr;
    }
    {
        auto* device_context = reinterpret_cast<AVHWDeviceContext*>(result->device->data);
        auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
        for (int index = 0; index < vulkan->nb_qf; ++index) {
            const auto flags = vulkan->qf[index].flags;
            if (flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
                result->prepare_queue_family = vulkan->qf[index].idx;
                break;
            }
        }
        for (int index = 0; index < vulkan->nb_qf; ++index) {
            const auto flags = vulkan->qf[index].flags;
            if ((flags & VK_QUEUE_COMPUTE_BIT) &&
                !(flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR)) {
                result->compute_queue_family = vulkan->qf[index].idx;
                break;
            }
        }
        if (result->compute_queue_family == VK_QUEUE_FAMILY_IGNORED) {
            for (int index = 0; index < vulkan->nb_qf; ++index) {
                if (vulkan->qf[index].flags & VK_QUEUE_COMPUTE_BIT) {
                    result->compute_queue_family = vulkan->qf[index].idx;
                    break;
                }
            }
        }
        if (result->prepare_queue_family == VK_QUEUE_FAMILY_IGNORED ||
            result->compute_queue_family == VK_QUEUE_FAMILY_IGNORED) {
            destroy_encoder(result);
            return nullptr;
        }
        VkDeviceQueueInfo2 queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2};
        queue_info.flags = vulkan->queue_flags;
        queue_info.queueFamilyIndex = result->prepare_queue_family;
        vkGetDeviceQueue2(vulkan->act_dev, &queue_info, &result->prepare_queue);
        queue_info.queueFamilyIndex = result->compute_queue_family;
        vkGetDeviceQueue2(vulkan->act_dev, &queue_info, &result->compute_queue);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = result->prepare_queue_family;
        if (result->prepare_queue == VK_NULL_HANDLE ||
            result->compute_queue == VK_NULL_HANDLE ||
            vkCreateCommandPool(vulkan->act_dev, &pool_info, vulkan->alloc, &result->prepare_pool) != VK_SUCCESS) {
            destroy_encoder(result);
            return nullptr;
        }
        pool_info.queueFamilyIndex = result->compute_queue_family;
        if (vkCreateCommandPool(vulkan->act_dev, &pool_info, vulkan->alloc, &result->compute_pool) != VK_SUCCESS) {
            destroy_encoder(result);
            return nullptr;
        }
    }

    // Separate single-plane RGBA producer pool. CUDA can import this image
    // format; the native bridge will later convert it to the NV12 Video image
    // on this same Vulkan device.
    result->rgba_frames = av_hwframe_ctx_alloc(result->device);
    if (!result->rgba_frames) {
        destroy_encoder(result);
        return nullptr;
    }
    auto* rgba_context = reinterpret_cast<AVHWFramesContext*>(result->rgba_frames->data);
    rgba_context->format = AV_PIX_FMT_VULKAN;
    rgba_context->sw_format = AV_PIX_FMT_RGBA;
    rgba_context->width = width;
    rgba_context->height = height;
    rgba_context->initial_pool_size = 3;
    auto* rgba_vulkan = reinterpret_cast<AVVulkanFramesContext*>(rgba_context->hwctx);
    rgba_vulkan->usage = static_cast<VkImageUsageFlagBits>(
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (av_hwframe_ctx_init(result->rgba_frames) < 0) {
        destroy_encoder(result);
        return nullptr;
    }

    VkVideoEncodeH264ProfileInfoKHR h264_profile{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR};
    VkVideoEncodeH265ProfileInfoKHR hevc_profile{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR};
    VkVideoProfileInfoKHR video_profile{VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR};
    VkVideoProfileListInfoKHR video_profiles{VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR};
    video_profile.videoCodecOperation = hevc
        ? VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR
        : VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
    video_profile.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
    video_profile.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
    video_profile.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
    if (hevc) {
        hevc_profile.stdProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
        video_profile.pNext = &hevc_profile;
    } else {
        h264_profile.stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
        video_profile.pNext = &h264_profile;
    }
    video_profiles.profileCount = 1;
    video_profiles.pProfiles = &video_profile;

    result->frames = av_hwframe_ctx_alloc(result->device);
    if (!result->frames) {
        destroy_encoder(result);
        return nullptr;
    }
    auto* frames_context = reinterpret_cast<AVHWFramesContext*>(result->frames->data);
    frames_context->format = AV_PIX_FMT_VULKAN;
    frames_context->sw_format = AV_PIX_FMT_NV12;
    frames_context->width = width;
    frames_context->height = height;
    frames_context->initial_pool_size = 3;
    auto* vulkan_frames = reinterpret_cast<AVVulkanFramesContext*>(frames_context->hwctx);
    // Keep one profile-compatible multi-plane NV12 image.  The native
    // Vulkan Compute stage writes intermediate Y/UV images and copies into
    // this image before Video Encode; split R8/R8G8 images are not legal
    // Vulkan Video input on the target NVIDIA driver.
    vulkan_frames->usage = static_cast<VkImageUsageFlagBits>(
        VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    vulkan_frames->flags = static_cast<AVVkFrameFlags>(0);
    vulkan_frames->create_pnext = &video_profiles;
    if (av_hwframe_ctx_init(result->frames) < 0 ||
        !initialize_convert_pipeline(result)) {
        destroy_encoder(result);
        return nullptr;
    }

    result->codec = avcodec_alloc_context3(codec);
    result->packet = av_packet_alloc();
    if (!result->codec || !result->packet) {
        destroy_encoder(result);
        return nullptr;
    }
    result->codec->width = width;
    result->codec->height = height;
    result->codec->time_base = AVRational{1, fps};
    result->codec->framerate = AVRational{fps, 1};
    result->codec->pix_fmt = AV_PIX_FMT_VULKAN;
    result->codec->bit_rate = target_bitrate > 0 ? target_bitrate : 40 * 1000 * 1000;
    result->codec->rc_max_rate = peak_bitrate > 0 ? peak_bitrate : result->codec->bit_rate;
    result->codec->gop_size = fps;
    result->codec->max_b_frames = 0;
    result->codec->hw_frames_ctx = av_buffer_ref(result->frames);
    if (!result->codec->hw_frames_ctx || avcodec_open2(result->codec, codec, nullptr) < 0) {
        destroy_encoder(result);
        return nullptr;
    }
    return result;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_acquire_frame(
    void* opaque, D2SVulkanVideoFrame* frame) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !frame) return -1;
    if (encoder->acquired) return -2;
    AVFrame* acquired = av_frame_alloc();
    if (!acquired || av_hwframe_get_buffer(encoder->frames, acquired, 0) < 0) {
        av_frame_free(&acquired);
        return -1;
    }
    encoder->acquired = acquired;
    auto* source = reinterpret_cast<AVVkFrame*>(acquired->data[0]);
    if (!prepare_frame_for_cuda(encoder, source)) {
        av_frame_free(&encoder->acquired);
        return -1;
    }
    copy_vk_frame(encoder, source, frame,
                  encoder->width, encoder->height);
    if (frame->plane_count > 0 && frame->external_handle_type)
        return 0;
    av_frame_free(&encoder->acquired);
    return -1;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_acquire_rgba_frame(
    void* opaque, D2SVulkanVideoFrame* frame) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !frame) return -1;
    if (encoder->acquired || encoder->rgba_acquired) return -2;
    AVFrame* acquired = av_frame_alloc();
    if (!acquired || av_hwframe_get_buffer(encoder->rgba_frames, acquired, 0) < 0) {
        av_frame_free(&acquired);
        return -1;
    }
    auto* source = reinterpret_cast<AVVkFrame*>(acquired->data[0]);
    if (!prepare_frame_for_cuda(encoder, source)) {
        av_frame_free(&acquired);
        return -1;
    }
    encoder->rgba_acquired = acquired;
    copy_vk_frame(encoder, source, frame, encoder->width, encoder->height, AV_PIX_FMT_RGBA);
    if (frame->plane_count != 1 || !frame->external_handle_type) {
        close_exported_handles(frame);
        av_frame_free(&encoder->rgba_acquired);
        return -1;
    }
    return 0;
}

bool submit_encode_acquire(Encoder* encoder, AVVkFrame* nv12,
                            unsigned long long wait_value,
                            unsigned long long* signal_value) {
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    const VkDevice device = vulkan->act_dev;
    const VkImage image = nv12->img[0];
    VkCommandBuffer command = VK_NULL_HANDLE;
    auto command_it = encoder->encode_acquire_buffers.find(image);
    auto fence_it = encoder->encode_acquire_fences.find(image);
    if (command_it == encoder->encode_acquire_buffers.end()) {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = encoder->prepare_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocate, &command) != VK_SUCCESS) return false;
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fence_info, vulkan->alloc, &fence) != VK_SUCCESS) return false;
        encoder->encode_acquire_buffers.emplace(image, command);
        encoder->encode_acquire_fences.emplace(image, fence);
    } else {
        command = command_it->second;
        fence_it = encoder->encode_acquire_fences.find(image);
        if (fence_it == encoder->encode_acquire_fences.end() ||
            vkWaitForFences(device, 1, &fence_it->second, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            vkResetFences(device, 1, &fence_it->second) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS) return false;
    }
    fence_it = encoder->encode_acquire_fences.find(image);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
    VkImageMemoryBarrier2 barriers[2]{};
    for (uint32_t index = 0; index < 2; ++index) {
        barriers[index] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barriers[index].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barriers[index].srcAccessMask = VK_ACCESS_2_NONE;
        barriers[index].dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
        barriers[index].dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
        barriers[index].oldLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
        barriers[index].newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
        barriers[index].srcQueueFamilyIndex = encoder->compute_queue_family;
        barriers[index].dstQueueFamilyIndex = encoder->prepare_queue_family;
        barriers[index].image = image;
        barriers[index].subresourceRange.aspectMask = index == 0
            ? VK_IMAGE_ASPECT_PLANE_0_BIT : VK_IMAGE_ASPECT_PLANE_1_BIT;
        barriers[index].subresourceRange.levelCount = 1;
        barriers[index].subresourceRange.layerCount = 1;
    }
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 2;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command, &dependency);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) return false;
    const unsigned long long next_value = wait_value + 1;
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = nv12->sem[0];
    wait.value = wait_value;
    wait.stageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = nv12->sem[0];
    signal.value = next_value;
    signal.stageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
    VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_info.commandBuffer = command;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command_info;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    auto submit2 = reinterpret_cast<PFN_vkQueueSubmit2>(
        vkGetDeviceProcAddr(device, "vkQueueSubmit2"));
    if (!submit2 || submit2(encoder->prepare_queue, 1, &submit, fence_it->second) != VK_SUCCESS)
        return false;
    *signal_value = next_value;
    return true;
}

bool submit_rgba_conversion(Encoder* encoder, AVVkFrame* rgba, AVVkFrame* nv12,
                             unsigned long long ready_value) {
    if (!encoder || !rgba || !nv12 || !ready_value || !rgba->sem[0] || !nv12->sem[0])
        return false;
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    const VkDevice device = vulkan->act_dev;
    const VkImage rgba_image = rgba->img[0];
    const VkImage command_key = encoder->convert_y_image;
    auto view_it = encoder->rgba_views.find(rgba_image);
    if (view_it == encoder->rgba_views.end()) {
        const VkImageView view = create_color_view(encoder, rgba_image, VK_FORMAT_R8G8B8A8_UNORM);
        if (!view) return false;
        view_it = encoder->rgba_views.emplace(rgba_image, view).first;
    }
    VkCommandBuffer command = VK_NULL_HANDLE;
    auto command_it = encoder->convert_buffers.find(command_key);
    auto fence_it = encoder->convert_fences.find(command_key);
    if (command_it == encoder->convert_buffers.end()) {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = encoder->compute_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocate, &command) != VK_SUCCESS) return false;
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(device, &fence_info, vulkan->alloc, &fence) != VK_SUCCESS) return false;
        encoder->convert_buffers.emplace(command_key, command);
        encoder->convert_fences.emplace(command_key, fence);
    } else {
        command = command_it->second;
        fence_it = encoder->convert_fences.find(command_key);
        if (fence_it == encoder->convert_fences.end() ||
            vkWaitForFences(device, 1, &fence_it->second, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            vkResetFences(device, 1, &fence_it->second) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS) return false;
    }
    fence_it = encoder->convert_fences.find(command_key);
    VkDescriptorImageInfo descriptor_infos[3]{};
    descriptor_infos[0] = {VK_NULL_HANDLE, view_it->second, VK_IMAGE_LAYOUT_GENERAL};
    descriptor_infos[1] = {VK_NULL_HANDLE, encoder->convert_y_view, VK_IMAGE_LAYOUT_GENERAL};
    descriptor_infos[2] = {VK_NULL_HANDLE, encoder->convert_uv_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t index = 0; index < 3; ++index) {
        writes[index] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[index].dstSet = encoder->convert_descriptor_set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[index].pImageInfo = &descriptor_infos[index];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
    VkImageMemoryBarrier2 barriers[6]{};
    uint32_t barrier_count = 0;
    auto add_barrier = [&](VkImage image, VkImageAspectFlags aspect, VkImageLayout old_layout,
                           VkImageLayout new_layout, VkPipelineStageFlags2 src_stage,
                           VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage,
                           VkAccessFlags2 dst_access) {
        auto& barrier = barriers[barrier_count++];
        barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = src_stage;
        barrier.srcAccessMask = src_access;
        barrier.dstStageMask = dst_stage;
        barrier.dstAccessMask = dst_access;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspect;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
    };
    add_barrier(rgba->img[0], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    barriers[0].srcQueueFamilyIndex = encoder->prepare_queue_family;
    barriers[0].dstQueueFamilyIndex = encoder->compute_queue_family;
    add_barrier(encoder->convert_y_image, VK_IMAGE_ASPECT_COLOR_BIT,
                encoder->convert_y_layout, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    add_barrier(encoder->convert_uv_image, VK_IMAGE_ASPECT_COLOR_BIT,
                encoder->convert_uv_layout, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    add_barrier(nv12->img[0], VK_IMAGE_ASPECT_PLANE_0_BIT,
                nv12->layout[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    barriers[3].srcQueueFamilyIndex = encoder->prepare_queue_family;
    barriers[3].dstQueueFamilyIndex = encoder->compute_queue_family;
    add_barrier(nv12->img[0], VK_IMAGE_ASPECT_PLANE_1_BIT,
                nv12->layout[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    barriers[4].srcQueueFamilyIndex = encoder->prepare_queue_family;
    barriers[4].dstQueueFamilyIndex = encoder->compute_queue_family;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = barrier_count;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command, &dependency);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, encoder->convert_pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            encoder->convert_pipeline_layout, 0, 1,
                            &encoder->convert_descriptor_set, 0, nullptr);
    vkCmdDispatch(command, (encoder->width + 7) / 8, (encoder->height + 7) / 8, 1);
    barrier_count = 0;
    add_barrier(encoder->convert_y_image, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    add_barrier(encoder->convert_uv_image, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    dependency.imageMemoryBarrierCount = barrier_count;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command, &dependency);
    VkImageCopy copies[2]{};
    copies[0].srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copies[0].dstSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1};
    copies[0].extent = {encoder->width, encoder->height, 1};
    copies[1].srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copies[1].dstSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1};
    copies[1].extent = {encoder->width / 2, encoder->height / 2, 1};
    vkCmdCopyImage(command, encoder->convert_y_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   nv12->img[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copies[0]);
    vkCmdCopyImage(command, encoder->convert_uv_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   nv12->img[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copies[1]);
    barrier_count = 0;
    add_barrier(nv12->img[0], VK_IMAGE_ASPECT_PLANE_0_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
    barriers[0].srcQueueFamilyIndex = encoder->compute_queue_family;
    barriers[0].dstQueueFamilyIndex = encoder->prepare_queue_family;
    add_barrier(nv12->img[0], VK_IMAGE_ASPECT_PLANE_1_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
    barriers[1].srcQueueFamilyIndex = encoder->compute_queue_family;
    barriers[1].dstQueueFamilyIndex = encoder->prepare_queue_family;
    add_barrier(rgba->img[0], VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
    barriers[2].srcQueueFamilyIndex = encoder->compute_queue_family;
    barriers[2].dstQueueFamilyIndex = encoder->prepare_queue_family;
    dependency.imageMemoryBarrierCount = barrier_count;
    dependency.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(command, &dependency);
    if (vkEndCommandBuffer(command) != VK_SUCCESS) return false;
    const unsigned long long signal_value = nv12->sem_value[0] + 1;
    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = rgba->sem[0];
    wait.value = ready_value;
    wait.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = nv12->sem[0];
    signal.value = signal_value;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    command_info.commandBuffer = command;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &command_info;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    auto submit2 = reinterpret_cast<PFN_vkQueueSubmit2>(
        vkGetDeviceProcAddr(device, "vkQueueSubmit2"));
    if (!submit2 || submit2(encoder->compute_queue, 1, &submit, fence_it->second) != VK_SUCCESS)
        return false;
    unsigned long long encode_value = 0;
    if (!submit_encode_acquire(encoder, nv12, signal_value, &encode_value)) return false;
    nv12->sem_value[0] = encode_value;
    encoder->convert_y_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    encoder->convert_uv_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    nv12->layout[0] = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR;
    nv12->access[0] = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
    nv12->queue_family[0] = encoder->prepare_queue_family;
    rgba->sem_value[0] = ready_value;
    rgba->layout[0] = VK_IMAGE_LAYOUT_GENERAL;
    rgba->access[0] = VK_ACCESS_2_MEMORY_READ_BIT;
    return true;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_release_rgba_frame(
    void* opaque, unsigned long long ready_value) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !encoder->rgba_acquired) return -1;
    auto* source = reinterpret_cast<AVVkFrame*>(encoder->rgba_acquired->data[0]);
    if (!source || !ready_value) return -2;
    for (unsigned int index = 0; index < 1 && source->sem[index]; ++index) {
        if (ready_value <= source->sem_value[index]) return -3;
        source->sem_value[index] = ready_value;
    }
    av_frame_free(&encoder->rgba_acquired);
    return 0;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_encode_rgba_frame(
    void* opaque, unsigned long long ready_value, long long timestamp) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !encoder->rgba_acquired || encoder->acquired || !ready_value)
        return -1;
    AVFrame* nv12 = av_frame_alloc();
    if (!nv12 || av_hwframe_get_buffer(encoder->frames, nv12, 0) < 0) {
        av_frame_free(&nv12);
        return -2;
    }
    auto* rgba = reinterpret_cast<AVVkFrame*>(encoder->rgba_acquired->data[0]);
    auto* destination = reinterpret_cast<AVVkFrame*>(nv12->data[0]);
    if (!submit_rgba_conversion(encoder, rgba, destination, ready_value)) {
        av_frame_free(&nv12);
        av_frame_free(&encoder->rgba_acquired);
        return -3;
    }
    nv12->pts = timestamp;
    const int result = avcodec_send_frame(encoder->codec, nv12);
    av_frame_free(&nv12);
    av_frame_free(&encoder->rgba_acquired);
    return result < 0 ? -4 : 0;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_submit_frame(
    void* opaque, const D2SVulkanVideoFrame* frame, void* ready_semaphore,
    unsigned long long ready_value, long long timestamp) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !encoder->acquired || !frame) return -1;
    // CUDA/HIP signals FFmpeg's exported per-plane timeline semaphore after
    // writing NV12. FFmpeg owns the VkSemaphore and waits on AVVkFrame's
    // sem_value during the Vulkan Video encode submission, so record the
    // producer's new value before handing the frame to avcodec.
    if (!ready_semaphore || ready_value == 0) {
        av_frame_free(&encoder->acquired);
        return -3;
    }
    if (frame->image[0] == nullptr || frame->plane_count == 0) {
        av_frame_free(&encoder->acquired);
        return -4;
    }
    auto* source = reinterpret_cast<AVVkFrame*>(encoder->acquired->data[0]);
    if (!source || !source->sem[0] || ready_value <= source->sem_value[0]) {
        av_frame_free(&encoder->acquired);
        return -5;
    }
    for (unsigned int index = 0; index < frame->plane_count; ++index) {
        if (!source->sem[index]) {
            av_frame_free(&encoder->acquired);
            return -5;
        }
        source->sem_value[index] = ready_value;
    }
    encoder->acquired->pts = timestamp;
    const int result = avcodec_send_frame(encoder->codec, encoder->acquired);
    av_frame_free(&encoder->acquired);
    return result < 0 ? -1 : 0;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_submit_image(
    void*, void*, void*, unsigned long long, long long) { return -2; }

extern "C" int d2s_vulkan_ffmpeg_encoder_read_packet(
    void* opaque, void* output, int capacity, long long* timestamp, int* keyframe) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !encoder->packet || !output || capacity < 1) return -1;
    const int result = avcodec_receive_packet(encoder->codec, encoder->packet);
    if (result < 0) return result == AVERROR(EAGAIN) ? 0 : -1;
    if (encoder->packet->size > capacity) return -2;
    std::memcpy(output, encoder->packet->data, static_cast<std::size_t>(encoder->packet->size));
    if (timestamp) *timestamp = encoder->packet->pts;
    if (keyframe) *keyframe = (encoder->packet->flags & AV_PKT_FLAG_KEY) != 0;
    const int size = encoder->packet->size;
    av_packet_unref(encoder->packet);
    return size;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_flush(void* opaque) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !encoder->codec) return -1;
    return avcodec_send_frame(encoder->codec, nullptr) < 0 ? -1 : 0;
}

extern "C" void d2s_vulkan_ffmpeg_encoder_destroy(void* opaque) {
    destroy_encoder(static_cast<Encoder*>(opaque));
}
