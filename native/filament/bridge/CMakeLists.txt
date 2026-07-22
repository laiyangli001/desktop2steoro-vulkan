cmake_minimum_required(VERSION 3.21)
project(filament_bridge LANGUAGES CXX)

if(WIN32)
    set(_filament_default_sdk "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/windows/v1.74.0")
elseif(APPLE)
    set(_filament_default_sdk "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/macos/v1.74.0/filament")
elseif(UNIX)
    set(_filament_default_sdk "${CMAKE_CURRENT_SOURCE_DIR}/../sdk/linux/v1.74.0/filament")
endif()
set(FILAMENT_SDK_ROOT "${_filament_default_sdk}" CACHE PATH "Root of the platform-specific Filament SDK")
if(NOT FILAMENT_SDK_ROOT)
    message(FATAL_ERROR "Set FILAMENT_SDK_ROOT to the extracted Filament SDK for this platform")
endif()
if(NOT EXISTS "${FILAMENT_SDK_ROOT}/include/filament/Engine.h")
    message(FATAL_ERROR "FILAMENT_SDK_ROOT does not contain include/filament/Engine.h")
endif()

# Filament's public VulkanPlatform header includes BlueVK, but release SDK
# archives do not ship that generated header. Keep the source revision pinned
# and materialize it only in the build tree.
set(_bluevk_include_dir "${CMAKE_CURRENT_BINARY_DIR}/bluevk/include")
set(_bluevk_header "${_bluevk_include_dir}/bluevk/BlueVK.h")
if(NOT EXISTS "${_bluevk_header}")
    file(MAKE_DIRECTORY "${_bluevk_include_dir}/bluevk")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/google/filament/v1.74.0/libs/bluevk/include/bluevk/BlueVK.h"
        "${_bluevk_header}"
        EXPECTED_HASH SHA256=6d8ad3dced6b1cfe93a57dde5f181b40265a280b1b4b660964964ab346a0456a
        STATUS _bluevk_download_status
        TLS_VERIFY ON)
    list(GET _bluevk_download_status 0 _bluevk_download_code)
    if(NOT _bluevk_download_code EQUAL 0)
        file(REMOVE "${_bluevk_header}")
        message(FATAL_ERROR "Unable to download pinned Filament BlueVK header: ${_bluevk_download_status}")
    endif()
endif()

find_path(D2S_VULKAN_INCLUDE_DIR vulkan/vulkan.h)
if(NOT D2S_VULKAN_INCLUDE_DIR)
    set(_vulkan_headers_root "${CMAKE_CURRENT_BINARY_DIR}/vulkan-headers")
    set(_vulkan_headers_archive "${CMAKE_CURRENT_BINARY_DIR}/vulkan-headers.tar.gz")
    if(NOT EXISTS "${_vulkan_headers_root}/include/vulkan/vulkan.h")
        file(DOWNLOAD
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.275.tar.gz"
            "${_vulkan_headers_archive}"
            SHOW_PROGRESS
            STATUS _vulkan_headers_download_status
            TLS_VERIFY ON)
        list(GET _vulkan_headers_download_status 0 _vulkan_headers_download_code)
        if(NOT _vulkan_headers_download_code EQUAL 0)
            message(FATAL_ERROR "Unable to download pinned Vulkan headers: ${_vulkan_headers_download_status}")
        endif()
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E tar xzf "${_vulkan_headers_archive}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            RESULT_VARIABLE _vulkan_headers_extract_code)
        if(NOT _vulkan_headers_extract_code EQUAL 0)
            message(FATAL_ERROR "Unable to extract pinned Vulkan headers")
        endif()
        file(GLOB _vulkan_headers_extracted LIST_DIRECTORIES true
            "${CMAKE_CURRENT_BINARY_DIR}/Vulkan-Headers-*")
        list(LENGTH _vulkan_headers_extracted _vulkan_headers_extracted_count)
        if(NOT _vulkan_headers_extracted_count EQUAL 1)
            message(FATAL_ERROR "Unexpected Vulkan headers archive layout")
        endif()
        file(RENAME "${_vulkan_headers_extracted}" "${_vulkan_headers_root}")
    endif()
    set(D2S_VULKAN_INCLUDE_DIR "${_vulkan_headers_root}/include")
endif()

add_library(filament_bridge SHARED
    filament_bridge.cpp
    bridge_context.cpp
    bridge_eye.cpp
    bridge_scene.cpp
    bridge_controller.cpp
    bridge_controller_guide.cpp
    bridge_laser.cpp
    bridge_screen.cpp
    bridge_material.cpp
    preview_bridge.cpp)
target_compile_features(filament_bridge PRIVATE cxx_std_20)
target_compile_definitions(filament_bridge PRIVATE FILAMENT_BRIDGE_BUILD=1)
if(MSVC)
    target_compile_options(filament_bridge PRIVATE /GR-)
else()
    target_compile_options(filament_bridge PRIVATE -fno-rtti)
endif()
target_include_directories(filament_bridge PRIVATE
    "${FILAMENT_SDK_ROOT}/include"
    "${_bluevk_include_dir}"
    "${D2S_VULKAN_INCLUDE_DIR}")

set(_filament_arch x86_64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(_filament_arch arm64)
endif()

if(WIN32)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}/md")
    # Filament's Windows static backend contains PlatformWGL symbols even
    # when the selected engine backend is Vulkan.
    target_link_libraries(filament_bridge PRIVATE user32 gdi32 shlwapi opengl32)
elseif(APPLE)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}")
    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    target_link_libraries(filament_bridge PRIVATE
        ${COCOA_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK})
elseif(UNIX)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}")
    target_link_libraries(filament_bridge PRIVATE dl pthread)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

file(GLOB _filament_libraries CONFIGURE_DEPENDS "${_filament_lib_dir}/*.lib" "${_filament_lib_dir}/*.a")
if(NOT _filament_libraries)
    message(FATAL_ERROR "No Filament libraries found in ${_filament_lib_dir}")
endif()
target_link_libraries(filament_bridge PRIVATE ${_filament_libraries})

if(WIN32)
    set(_filament_output_platform windows)
elseif(APPLE)
    set(_filament_output_platform macos)
elseif(UNIX)
    set(_filament_output_platform linux)
endif()
set(_filament_output_dir
    "${CMAKE_SOURCE_DIR}/../../../src/xr_viewer/native/${_filament_output_platform}")
file(MAKE_DIRECTORY "${_filament_output_dir}")
set_target_properties(filament_bridge PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    OUTPUT_NAME filament_bridge
    RUNTIME_OUTPUT_DIRECTORY "${_filament_output_dir}"
    LIBRARY_OUTPUT_DIRECTORY "${_filament_output_dir}")

foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(filament_bridge PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${_config} "${_filament_output_dir}"
        LIBRARY_OUTPUT_DIRECTORY_${_config} "${_filament_output_dir}")
endforeach()
