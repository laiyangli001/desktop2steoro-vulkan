#include "vulkan_ffmpeg_bridge.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
}

#include <cstdio>
#include <cstring>

namespace {
void write_message(char* output, int capacity, const char* message) {
    if (!output || capacity <= 0) return;
    std::snprintf(output, static_cast<std::size_t>(capacity), "%s", message);
}
}

extern "C" int d2s_vulkan_ffmpeg_bridge_abi_version() { return 1; }

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
        "FFmpeg Vulkan device and encoder discovery passed; image submit ABI pending");
    return 1;
}

extern "C" void* d2s_vulkan_ffmpeg_encoder_create(
    void*, void*, void*, void*, int, int, int, int, int, int, int) {
    return nullptr;
}

extern "C" int d2s_vulkan_ffmpeg_encoder_submit_image(
    void*, void*, void*, unsigned long long, long long) { return -1; }

extern "C" int d2s_vulkan_ffmpeg_encoder_read_packet(
    void*, void*, int, long long*, int*) { return -1; }

extern "C" void d2s_vulkan_ffmpeg_encoder_destroy(void*) {}
