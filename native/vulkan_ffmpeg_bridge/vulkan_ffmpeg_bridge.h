#pragma once

#ifdef _WIN32
#define D2S_VULKAN_FFMPEG_API __declspec(dllexport)
#else
#define D2S_VULKAN_FFMPEG_API
#endif

// Narrow ABI for the eventual in-process FFmpeg Vulkan encoder. Handles are
// opaque across the Python/native boundary; the native side owns AVCodec,
// AVHWFramesContext and packet lifetime.
extern "C" {

D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_bridge_abi_version();
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_bridge_probe(char* output, int capacity);
D2S_VULKAN_FFMPEG_API void* d2s_vulkan_ffmpeg_encoder_create(
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
    int hevc);

// Submit an already synchronized Vulkan encode-source image. The image must
// be in VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR when this call is made. No CPU
// pixel pointer is accepted by this ABI.
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_submit_image(
    void* encoder,
    void* vk_image,
    void* ready_semaphore,
    unsigned long long ready_value,
    long long timestamp);
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_read_packet(
    void* encoder,
    void* output,
    int capacity,
    long long* timestamp,
    int* keyframe);
D2S_VULKAN_FFMPEG_API void d2s_vulkan_ffmpeg_encoder_destroy(void* encoder);

}

