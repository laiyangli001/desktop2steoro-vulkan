cmake_minimum_required(VERSION 3.21)
project(d2s_nvenc_cudaarray_bridge LANGUAGES CXX CUDA)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_nvenc_cudaarray_bridge is currently implemented for Windows only")
endif()

set(D2S_NV_CODEC_HEADERS "" CACHE PATH "NVENC API headers root")
if(NOT EXISTS "${D2S_NV_CODEC_HEADERS}/include/ffnvcodec/nvEncodeAPI.h")
    message(FATAL_ERROR "Set D2S_NV_CODEC_HEADERS to the NVENC API headers root")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(d2s_nvenc_cudaarray_bridge SHARED
    nvenc_cudaarray_bridge.cpp
    cudaarray_surface_writer.cu
)
target_compile_features(d2s_nvenc_cudaarray_bridge PRIVATE cxx_std_17)
target_compile_options(d2s_nvenc_cudaarray_bridge PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
)
set_target_properties(d2s_nvenc_cudaarray_bridge PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "61;75;86;89"
)
target_compile_definitions(d2s_nvenc_cudaarray_bridge PRIVATE
    D2S_NVENC_CUDAARRAY_BUILD=1
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)
target_include_directories(d2s_nvenc_cudaarray_bridge PRIVATE
    "${D2S_NV_CODEC_HEADERS}/include/ffnvcodec"
)
target_link_libraries(d2s_nvenc_cudaarray_bridge PRIVATE CUDA::cudart)
set_target_properties(d2s_nvenc_cudaarray_bridge PROPERTIES
    OUTPUT_NAME "d2s_nvenc_cudaarray_bridge"
)
