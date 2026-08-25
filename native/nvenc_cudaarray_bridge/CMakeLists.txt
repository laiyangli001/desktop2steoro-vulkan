cmake_minimum_required(VERSION 3.21)
project(d2s_nvenc_cudaarray_bridge LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_nvenc_cudaarray_bridge is currently implemented for Windows only")
endif()

set(D2S_NV_CODEC_HEADERS "" CACHE PATH "nv-codec-headers root")
if(NOT EXISTS "${D2S_NV_CODEC_HEADERS}/include/ffnvcodec/nvEncodeAPI.h")
    message(FATAL_ERROR "Set D2S_NV_CODEC_HEADERS to the nv-codec-headers repository root")
endif()

add_library(d2s_nvenc_cudaarray_bridge SHARED
    nvenc_cudaarray_bridge.cpp
)
target_compile_features(d2s_nvenc_cudaarray_bridge PRIVATE cxx_std_17)
target_compile_definitions(d2s_nvenc_cudaarray_bridge PRIVATE
    D2S_NVENC_CUDAARRAY_BUILD=1
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)
target_include_directories(d2s_nvenc_cudaarray_bridge PRIVATE
    "${D2S_NV_CODEC_HEADERS}/include/ffnvcodec"
)
set_target_properties(d2s_nvenc_cudaarray_bridge PROPERTIES
    OUTPUT_NAME "d2s_nvenc_cudaarray_bridge"
)
