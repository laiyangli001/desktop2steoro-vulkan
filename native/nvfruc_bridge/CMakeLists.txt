cmake_minimum_required(VERSION 3.21)
project(d2s_nvfruc_bridge LANGUAGES CXX CUDA)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_nvfruc_bridge is currently implemented for Windows only")
endif()

set(D2S_NVFRUC_SDK_ROOT "" CACHE PATH "NVIDIA Optical Flow SDK root containing NvOFFRUC.h and NvOFFRUC.dll")
if(EXISTS "${D2S_NVFRUC_SDK_ROOT}/include/NvOFFRUC.h")
    set(D2S_NVFRUC_INCLUDE_DIR "${D2S_NVFRUC_SDK_ROOT}/include")
elseif(EXISTS "${D2S_NVFRUC_SDK_ROOT}/Interface/NvOFFRUC.h")
    set(D2S_NVFRUC_INCLUDE_DIR "${D2S_NVFRUC_SDK_ROOT}/Interface")
else()
    message(FATAL_ERROR "Set D2S_NVFRUC_SDK_ROOT to an NVIDIA Optical Flow SDK root containing NvOFFRUC.h")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(d2s_nvfruc_bridge SHARED nvfruc_bridge.cpp)
target_compile_features(d2s_nvfruc_bridge PRIVATE cxx_std_17)
target_compile_definitions(d2s_nvfruc_bridge PRIVATE
    D2S_NVFRUC_BUILD=1
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)
target_include_directories(d2s_nvfruc_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${D2S_NVFRUC_INCLUDE_DIR}"
)
target_link_libraries(d2s_nvfruc_bridge PRIVATE CUDA::cudart)
set_target_properties(d2s_nvfruc_bridge PROPERTIES
    OUTPUT_NAME "d2s_nvfruc_bridge"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
)
