cmake_minimum_required(VERSION 3.20)
project(Desktop2StereoMacLauncher LANGUAGES OBJCXX)

add_executable(Desktop2Stereo MACOSX_BUNDLE main.mm)
set_target_properties(Desktop2Stereo PROPERTIES
    OUTPUT_NAME "Desktop2Stereo"
    MACOSX_BUNDLE_GUI_IDENTIFIER "site.d2s.desktop2stereo.launcher"
    MACOSX_BUNDLE_BUNDLE_NAME "Desktop2Stereo"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist")
target_compile_features(Desktop2Stereo PRIVATE cxx_std_17)
target_link_libraries(Desktop2Stereo PRIVATE "-framework AppKit" "-framework Foundation")
