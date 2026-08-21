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
#include <memory>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
struct Encoder {
    AVBufferRef* device = nullptr;
    AVBufferRef* frames = nullptr;
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* acquired = nullptr;
    int width = 0;
    int height = 0;
    int fps = 0;
    VkQueue prepare_queue = VK_NULL_HANDLE;
    uint32_t prepare_queue_family = VK_QUEUE_FAMILY_IGNORED;
    VkCommandPool prepare_pool = VK_NULL_HANDLE;
    std::unordered_map<AVVkFrame*, VkCommandBuffer> prepare_buffers;
};

constexpr unsigned int D2S_EXTERNAL_HANDLE_WIN32 = 1;
constexpr unsigned int D2S_EXTERNAL_HANDLE_FD = 2;

void write_message(char* output, int capacity, const char* message) {
    if (!output || capacity <= 0) return;
    std::snprintf(output, static_cast<std::size_t>(capacity), "%s", message);
}

void destroy_encoder(Encoder* encoder) {
    if (!encoder) return;
    av_frame_free(&encoder->acquired);
    av_packet_free(&encoder->packet);
    avcodec_free_context(&encoder->codec);
    av_buffer_unref(&encoder->frames);
    if (encoder->prepare_pool != VK_NULL_HANDLE && encoder->device) {
        auto* device_context = reinterpret_cast<AVHWDeviceContext*>(encoder->device->data);
        auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
        vkDeviceWaitIdle(vulkan->act_dev);
        vkDestroyCommandPool(vulkan->act_dev, encoder->prepare_pool, vulkan->alloc);
    }
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
                   int width, int height) {
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
    const VkFormat* formats = av_vkfmt_from_pixfmt(AV_PIX_FMT_NV12);
    if (formats) {
        for (unsigned int index = 0; index < destination->plane_count; ++index)
            destination->format[index] = static_cast<unsigned int>(formats[index]);
    }
    if (destination->plane_count > 0 && !export_vk_frame_handles(encoder, source, destination))
        close_exported_handles(destination);
}
}

extern "C" int d2s_vulkan_ffmpeg_bridge_abi_version() { return 3; }

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
            if (vulkan->qf[index].flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
                result->prepare_queue_family = vulkan->qf[index].idx;
                break;
            }
        }
        if (result->prepare_queue_family == VK_QUEUE_FAMILY_IGNORED && vulkan->nb_qf)
            result->prepare_queue_family = vulkan->qf[0].idx;
        if (result->prepare_queue_family == VK_QUEUE_FAMILY_IGNORED) {
            destroy_encoder(result);
            return nullptr;
        }
        vkGetDeviceQueue(vulkan->act_dev, result->prepare_queue_family, 0, &result->prepare_queue);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = result->prepare_queue_family;
        if (result->prepare_queue == VK_NULL_HANDLE ||
            vkCreateCommandPool(vulkan->act_dev, &pool_info, vulkan->alloc, &result->prepare_pool) != VK_SUCCESS) {
            destroy_encoder(result);
            return nullptr;
        }
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
    // CUDA external-memory arrays do not require STORAGE usage.  Request only
    // the Video Encode source usage guaranteed by the selected profile; FFmpeg
    // still adds optional transfer/storage usages when the driver supports them.
    vulkan_frames->usage = static_cast<VkImageUsageFlagBits>(
        VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR);
    vulkan_frames->flags = AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE;
    vulkan_frames->create_pnext = &video_profiles;
    if (av_hwframe_ctx_init(result->frames) < 0) {
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
