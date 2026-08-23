cmake_minimum_required(VERSION 3.20)
project(d2s_d3d11_sbs_surface LANGUAGES CXX)

add_library(d2s_d3d11_sbs_surface SHARED bridge.cpp)
target_compile_features(d2s_d3d11_sbs_surface PRIVATE cxx_std_17)
target_include_directories(d2s_d3d11_sbs_surface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)
    target_link_libraries(d2s_d3d11_sbs_surface PRIVATE d3d11 dxgi)
endif()
