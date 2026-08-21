#include "vulkan_ffmpeg_bridge.h"

#include <vulkan/vulkan.h>

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
};

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
    av_buffer_unref(&encoder->device);
    delete encoder;
}

void copy_vk_frame(const AVVkFrame* source, D2SVulkanVideoFrame* destination,
                   int width, int height) {
    std::memset(destination, 0, sizeof(*destination));
    destination->width = static_cast<unsigned int>(width);
    destination->height = static_cast<unsigned int>(height);
    destination->plane_count = 0;
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
}
}

extern "C" int d2s_vulkan_ffmpeg_bridge_abi_version() { return 2; }

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
    if (!vk_instance || !vk_physical_device || !vk_device || !vk_queue ||
        width < 1 || height < 1 || fps < 1)
        return nullptr;

    const char* encoder_name = hevc ? "hevc_vulkan" : "h264_vulkan";
    const AVCodec* codec = avcodec_find_encoder_by_name(encoder_name);
    if (!codec) return nullptr;

    auto* result = new Encoder();
    result->width = width;
    result->height = height;
    result->fps = fps;
    result->device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
    if (!result->device) {
        destroy_encoder(result);
        return nullptr;
    }
    auto* device_context = reinterpret_cast<AVHWDeviceContext*>(result->device->data);
    auto* vulkan = reinterpret_cast<AVVulkanDeviceContext*>(device_context->hwctx);
    vulkan->inst = reinterpret_cast<VkInstance>(vk_instance);
    vulkan->phys_dev = reinterpret_cast<VkPhysicalDevice>(vk_physical_device);
    vulkan->act_dev = reinterpret_cast<VkDevice>(vk_device);
    vulkan->qf[0].idx = queue_family;
    vulkan->qf[0].num = 1;
    vulkan->qf[0].flags = static_cast<VkQueueFlagBits>(
        VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    vulkan->nb_qf = 1;
    if (av_hwdevice_ctx_init(result->device) < 0) {
        destroy_encoder(result);
        return nullptr;
    }

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
    vulkan_frames->usage = static_cast<VkImageUsageFlagBits>(
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR);
    vulkan_frames->flags = AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE;
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
    copy_vk_frame(reinterpret_cast<AVVkFrame*>(acquired->data[0]), frame,
                  encoder->width, encoder->height);
    return frame->plane_count > 0 ? 0 : -1;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_submit_frame(
    void* opaque, const D2SVulkanVideoFrame* frame, void* ready_semaphore,
    unsigned long long ready_value, long long timestamp) {
    auto* encoder = static_cast<Encoder*>(opaque);
    if (!encoder || !encoder->acquired || !frame) return -1;
    // The frame-pool ABI must not silently accept a producer that has not
    // published a completion point.  Queue submission for this point is
    // intentionally kept in the native bridge; until it is implemented the
    // call fails closed and the caller must use the stable host-upload path.
    if (!ready_semaphore || ready_value == 0) {
        av_frame_free(&encoder->acquired);
        return -3;
    }
    if (frame->image[0] == nullptr || frame->plane_count == 0) {
        av_frame_free(&encoder->acquired);
        return -4;
    }
    // TODO: import/wait the external semaphore on the shared Vulkan queue and
    // transition the FFmpeg-owned images to VIDEO_ENCODE_SRC before calling
    // avcodec_send_frame.  Do not remove this guard until that wait is real.
    av_frame_free(&encoder->acquired);
    return -3;
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
