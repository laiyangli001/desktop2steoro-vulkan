cmake_minimum_required(VERSION 3.20)
project(Desktop2StereoLinuxLauncher LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(X11 REQUIRED IMPORTED_TARGET x11)
pkg_check_modules(PNG REQUIRED IMPORTED_TARGET libpng)

add_executable(Desktop2Stereo main.cpp)
target_compile_features(Desktop2Stereo PRIVATE cxx_std_17)
target_link_libraries(Desktop2Stereo PRIVATE PkgConfig::X11 PkgConfig::PNG)
set_target_properties(Desktop2Stereo PROPERTIES OUTPUT_NAME "Desktop2Stereo")
