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

add_library(filament_bridge SHARED filament_bridge.cpp)
target_compile_features(filament_bridge PRIVATE cxx_std_20)
target_compile_definitions(filament_bridge PRIVATE FILAMENT_BRIDGE_BUILD=1)
target_include_directories(filament_bridge PRIVATE "${FILAMENT_SDK_ROOT}/include")

set(_filament_arch x86_64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(_filament_arch arm64)
endif()

if(WIN32)
    set(_filament_lib_dir "${FILAMENT_SDK_ROOT}/lib/${_filament_arch}/md")
    target_link_libraries(filament_bridge PRIVATE user32 gdi32 shlwapi)
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

set(_filament_output_dir "${CMAKE_SOURCE_DIR}/../../../src/xr_viewer/native")
set_target_properties(filament_bridge PROPERTIES
    OUTPUT_NAME filament_bridge
    RUNTIME_OUTPUT_DIRECTORY "${_filament_output_dir}"
    LIBRARY_OUTPUT_DIRECTORY "${_filament_output_dir}")

foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(filament_bridge PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${_config} "${_filament_output_dir}"
        LIBRARY_OUTPUT_DIRECTORY_${_config} "${_filament_output_dir}")
endforeach()
