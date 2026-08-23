cmake_minimum_required(VERSION 3.21)
project(d2s_onevpl_d3d11_encoder LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_onevpl_d3d11_encoder is currently implemented for Windows only")
endif()

add_library(d2s_onevpl_d3d11_encoder SHARED onevpl_d3d11_encoder.cpp)
target_compile_features(d2s_onevpl_d3d11_encoder PRIVATE cxx_std_17)
target_compile_definitions(d2s_onevpl_d3d11_encoder PRIVATE
    D2S_ONEVPL_BUILD=1
    WIN32_LEAN_AND_MEAN
    NOMINMAX)
target_include_directories(d2s_onevpl_d3d11_encoder PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(d2s_onevpl_d3d11_encoder PRIVATE d3d11 dxgi)

set(D2S_ONEVPL_ROOT "" CACHE PATH
    "oneVPL SDK root containing include and lib")
if(D2S_ONEVPL_ROOT)
    find_path(D2S_ONEVPL_INCLUDE_DIR vpl/mfx.h
        PATHS "${D2S_ONEVPL_ROOT}/include" NO_DEFAULT_PATH)
    find_library(D2S_ONEVPL_LIBRARY NAMES vpl
        PATHS "${D2S_ONEVPL_ROOT}/lib"
              "${D2S_ONEVPL_ROOT}/lib/x64"
              "${D2S_ONEVPL_ROOT}/runtime/lib/intel64"
        NO_DEFAULT_PATH)
endif()

if(D2S_ONEVPL_INCLUDE_DIR AND D2S_ONEVPL_LIBRARY)
    target_compile_definitions(d2s_onevpl_d3d11_encoder PRIVATE D2S_ONEVPL_SDK=1)
    target_include_directories(d2s_onevpl_d3d11_encoder PRIVATE
        "${D2S_ONEVPL_INCLUDE_DIR}")
    target_link_libraries(d2s_onevpl_d3d11_encoder PRIVATE
        "${D2S_ONEVPL_LIBRARY}")
else()
    message(STATUS
        "oneVPL SDK not configured; building a safe unavailable stub")
endif()

set_target_properties(d2s_onevpl_d3d11_encoder PROPERTIES
    OUTPUT_NAME "d2s_onevpl_d3d11_encoder"
    PREFIX "")
