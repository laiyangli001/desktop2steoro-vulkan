cmake_minimum_required(VERSION 3.21)
project(d2s_vulkan_ffmpeg_bridge LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(D2S_FFMPEG_ROOT "" CACHE PATH "FFmpeg package containing include and lib")
if(NOT D2S_FFMPEG_ROOT)
    message(FATAL_ERROR "Set D2S_FFMPEG_ROOT to the extracted FFmpeg runtime/development package")
endif()
if(NOT EXISTS "${D2S_FFMPEG_ROOT}/include/libavcodec/avcodec.h")
    message(FATAL_ERROR "D2S_FFMPEG_ROOT does not contain libavcodec headers")
endif()

add_library(d2s_vulkan_ffmpeg_bridge SHARED bridge.cpp)
target_compile_features(d2s_vulkan_ffmpeg_bridge PRIVATE cxx_std_20)
target_compile_definitions(d2s_vulkan_ffmpeg_bridge PRIVATE D2S_VULKAN_FFMPEG_BRIDGE_BUILD=1)
target_include_directories(d2s_vulkan_ffmpeg_bridge PRIVATE
    "${D2S_FFMPEG_ROOT}/include"
    "${Vulkan_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(d2s_vulkan_ffmpeg_bridge PRIVATE Vulkan::Vulkan)

find_library(D2S_AVCODEC_LIBRARY NAMES avcodec-63 avcodec libavcodec
    PATHS "${D2S_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(D2S_AVUTIL_LIBRARY NAMES avutil-61 avutil libavutil
    PATHS "${D2S_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(d2s_vulkan_ffmpeg_bridge PRIVATE
    "${D2S_AVCODEC_LIBRARY}"
    "${D2S_AVUTIL_LIBRARY}")

set_target_properties(d2s_vulkan_ffmpeg_bridge PROPERTIES
    OUTPUT_NAME d2s_vulkan_ffmpeg_bridge
    PREFIX "")
