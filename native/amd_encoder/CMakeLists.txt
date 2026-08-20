cmake_minimum_required(VERSION 3.21)
project(d2s_amd_encoder LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_amd_encoder is currently implemented for Windows only")
endif()

add_library(d2s_amd_encoder SHARED amd_encoder.cpp)
target_compile_features(d2s_amd_encoder PRIVATE cxx_std_17)
target_compile_definitions(d2s_amd_encoder PRIVATE D2S_AMD_ENCODER_BUILD=1 WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(d2s_amd_encoder PRIVATE d3d11 dxgi)
set_target_properties(d2s_amd_encoder PROPERTIES OUTPUT_NAME "d2s_amd_encoder")

