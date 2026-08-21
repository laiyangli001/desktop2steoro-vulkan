#pragma once

#ifdef _WIN32
#define D2S_VULKAN_FFMPEG_API __declspec(dllexport)
#else
#define D2S_VULKAN_FFMPEG_API
#endif

// Narrow ABI for the in-process FFmpeg Vulkan encoder. Handles are opaque
// across the Python/native boundary; the native side owns AVCodec,
// AVHWFramesContext and packet lifetime. The frame pool is allocated by
// FFmpeg/Vulkan and exposes only GPU image handles; no CPU pixel pointer is
// part of this ABI.
extern "C" {

D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_bridge_abi_version();
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_bridge_probe(char* output, int capacity);

typedef struct D2SVulkanVideoFrame {
    void* image[2];
    void* memory[2];
    unsigned long long memory_size[2];
    long long memory_offset[2];
    unsigned int format[2];
    unsigned int layout[2];
    unsigned int width;
    unsigned int height;
    unsigned int plane_count;
    // OS handles duplicated for this acquire. The caller imports each handle
    // into CUDA/HIP once and closes it immediately after that import. They are
    // never CPU pixel buffers. On Windows these are HANDLE values; on Linux
    // they are file descriptors stored as signed 64-bit values.
    long long external_memory_handle[2];
    long long external_semaphore_handle[2];
    unsigned long long semaphore_value[2];
    unsigned long long slot_id;
    unsigned int external_handle_type;
} D2SVulkanVideoFrame;

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

// Acquire one FFmpeg-owned NV12 Vulkan frame from the bounded pool. The
// caller writes the returned GPU images with Vulkan/CUDA interop and then
// submits the same descriptor. The producer-ready semaphore/value are
// mandatory; the bridge records that timeline value on FFmpeg's AVVkFrame so
// Vulkan Video waits for GPU writes before encoding. Return 0 on success, -1
// on failure, -2 when the previous frame has not been submitted yet.
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_acquire_frame(
    void* encoder,
    D2SVulkanVideoFrame* frame);

// Acquire one single-plane RGBA image on the same FFmpeg-owned Vulkan device.
// CUDA writes this image through the exported external handles; it is never
// submitted directly to Vulkan Video. release_rgba_frame returns the slot to
// the RGBA pool after the producer has finished its GPU work.
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_acquire_rgba_frame(
    void* encoder,
    D2SVulkanVideoFrame* frame);
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_release_rgba_frame(
    void* encoder);

D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_submit_frame(
    void* encoder,
    const D2SVulkanVideoFrame* frame,
    void* ready_semaphore,
    unsigned long long ready_value,
    long long timestamp);

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
D2S_VULKAN_FFMPEG_API int d2s_vulkan_ffmpeg_encoder_flush(void* encoder);
D2S_VULKAN_FFMPEG_API void d2s_vulkan_ffmpeg_encoder_destroy(void* encoder);

}
