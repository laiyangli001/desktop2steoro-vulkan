cmake_minimum_required(VERSION 3.20)
project(Desktop2StereoMacLauncher LANGUAGES OBJCXX)

add_executable(Desktop2Stereo main.mm)
set_target_properties(Desktop2Stereo PROPERTIES
    OUTPUT_NAME "Desktop2Stereo-macos"
    CXX_STANDARD 17
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED YES
    MACOSX_RPATH ON)
target_compile_features(Desktop2Stereo PRIVATE cxx_std_17)
target_link_libraries(Desktop2Stereo PRIVATE "-framework AppKit" "-framework Foundation")
