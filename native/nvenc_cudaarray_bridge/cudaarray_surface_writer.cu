#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>

namespace {

__device__ __forceinline__ unsigned char normalized_u8(float value) {
    value = fminf(1.0f, fmaxf(0.0f, value));
    return static_cast<unsigned char>(value * 255.0f + 0.5f);
}

__device__ __forceinline__ unsigned char read_channel(
    const unsigned char* address,
    int32_t scalar_type
) {
    if (scalar_type == 0) {
        return *address;
    }
    if (scalar_type == 1) {
        return normalized_u8(*reinterpret_cast<const float*>(address));
    }
    if (scalar_type == 2) {
        return normalized_u8(
            __half2float(*reinterpret_cast<const __half*>(address))
        );
    }
    return 0;
}

__global__ void write_rgba_surface(
    cudaSurfaceObject_t surface,
    const unsigned char* source,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    size_t stride_y,
    size_t stride_x,
    size_t stride_c,
    int32_t scalar_type
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* pixel =
        source + static_cast<size_t>(y) * stride_y +
        static_cast<size_t>(x) * stride_x;
    uchar4 rgba;
    if (channels == 1) {
        const unsigned char value = read_channel(pixel, scalar_type);
        rgba = make_uchar4(value, value, value, 255);
    } else {
        rgba.x = read_channel(pixel, scalar_type);
        rgba.y = read_channel(pixel + stride_c, scalar_type);
        rgba.z = read_channel(pixel + stride_c * 2, scalar_type);
        rgba.w = channels >= 4
            ? read_channel(pixel + stride_c * 3, scalar_type)
            : 255;
    }
    surf2Dwrite(rgba, surface, x * sizeof(uchar4), y);
}

}  // namespace

extern "C" int32_t d2s_cudaarray_surface_create(
    void* cuda_array,
    uint64_t* surface
) {
    if (!cuda_array || !surface) {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }
    cudaResourceDesc resource;
    memset(&resource, 0, sizeof(resource));
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = static_cast<cudaArray_t>(cuda_array);
    cudaSurfaceObject_t object = 0;
    const cudaError_t status = cudaCreateSurfaceObject(&object, &resource);
    if (status == cudaSuccess) {
        *surface = static_cast<uint64_t>(object);
    }
    return static_cast<int32_t>(status);
}

extern "C" int32_t d2s_cudaarray_surface_write(
    uint64_t surface,
    uint64_t device_pointer,
    uint32_t width,
    uint32_t height,
    uint32_t channels,
    size_t stride_y,
    size_t stride_x,
    size_t stride_c,
    int32_t scalar_type,
    uint64_t cuda_stream
) {
    if (
        surface == 0 ||
        device_pointer == 0 ||
        width == 0 ||
        height == 0 ||
        (channels != 1 && channels != 3 && channels != 4) ||
        scalar_type < 0 ||
        scalar_type > 2
    ) {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }

    const dim3 block(16, 16);
    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    const auto stream = reinterpret_cast<cudaStream_t>(
        static_cast<uintptr_t>(cuda_stream)
    );
    write_rgba_surface<<<grid, block, 0, stream>>>(
        static_cast<cudaSurfaceObject_t>(surface),
        reinterpret_cast<const unsigned char*>(
            static_cast<uintptr_t>(device_pointer)
        ),
        width,
        height,
        channels,
        stride_y,
        stride_x,
        stride_c,
        scalar_type
    );
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return static_cast<int32_t>(status);
    }
    // NVENC submits work through a separate queue and has no implicit
    // dependency on the PyTorch stream that produced the SBS tensor.
    status = cudaStreamSynchronize(stream);
    return static_cast<int32_t>(status);
}

extern "C" int32_t d2s_cudaarray_surface_destroy(uint64_t surface) {
    if (surface == 0) {
        return static_cast<int32_t>(cudaSuccess);
    }
    return static_cast<int32_t>(
        cudaDestroySurfaceObject(static_cast<cudaSurfaceObject_t>(surface))
    );
}

extern "C" const char* d2s_cudaarray_surface_error(int32_t status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
