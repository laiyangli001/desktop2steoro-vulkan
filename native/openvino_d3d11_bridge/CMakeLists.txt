cmake_minimum_required(VERSION 3.21)
project(d2s_openvino_d3d11_bridge LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "d2s_openvino_d3d11_bridge is currently implemented for Windows only")
endif()

set(D2S_OPENVINO_ROOT "" CACHE PATH "OpenVINO SDK root containing include and lib")
if(NOT D2S_OPENVINO_ROOT)
    message(FATAL_ERROR "Set D2S_OPENVINO_ROOT to an OpenVINO SDK root")
endif()

find_path(D2S_OPENVINO_INCLUDE_DIR openvino/openvino.hpp
    PATHS "${D2S_OPENVINO_ROOT}/include"
          "${D2S_OPENVINO_ROOT}/runtime/include"
    NO_DEFAULT_PATH REQUIRED)
find_library(D2S_OPENVINO_LIBRARY NAMES openvino
    PATHS "${D2S_OPENVINO_ROOT}/lib"
          "${D2S_OPENVINO_ROOT}/lib/intel64"
          "${D2S_OPENVINO_ROOT}/runtime/lib/intel64"
    PATH_SUFFIXES Release Debug
    NO_DEFAULT_PATH REQUIRED)

add_library(d2s_openvino_d3d11_bridge SHARED bridge.cpp)
target_compile_features(d2s_openvino_d3d11_bridge PRIVATE cxx_std_17)
target_compile_definitions(d2s_openvino_d3d11_bridge PRIVATE D2S_OPENVINO_D3D11_BUILD=1 WIN32_LEAN_AND_MEAN NOMINMAX)
target_include_directories(d2s_openvino_d3d11_bridge PRIVATE
    "${D2S_OPENVINO_INCLUDE_DIR}")
target_link_libraries(d2s_openvino_d3d11_bridge PRIVATE
    "${D2S_OPENVINO_LIBRARY}"
    d3d11
    dxgi)
set_target_properties(d2s_openvino_d3d11_bridge PROPERTIES
    OUTPUT_NAME "d2s_openvino_d3d11_bridge"
    PREFIX "")
