from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SOURCE = ROOT / "native" / "launcher" / "windows" / "main.cpp"
WINDOWS_CMAKE = ROOT / "native" / "launcher" / "windows" / "CMakeLists.txt"
RUN_WINDOWS = ROOT / "src" / "run_windows.bat"
LINUX_SOURCE = ROOT / "native" / "launcher" / "linux" / "main.cpp"
LINUX_CMAKE = ROOT / "native" / "launcher" / "linux" / "CMakeLists.txt"
MACOS_SOURCE = ROOT / "native" / "launcher" / "macos" / "main.mm"
MACOS_CMAKE = ROOT / "native" / "launcher" / "macos" / "CMakeLists.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "build-native-launcher.yml"


def test_windows_launcher_uses_native_layered_splash_and_ready_handshake():
    source = WINDOWS_SOURCE.read_text(encoding="utf-8")
    assert "WS_EX_LAYERED" in source
    assert "UpdateLayeredWindow" in source
    assert "CLSID_WICImagingFactory" in source
    assert "gui_ready.flag" in source
    assert "CreateProcessW" in source


def test_windows_launcher_build_definition_is_native_executable():
    cmake = WINDOWS_CMAKE.read_text(encoding="utf-8")
    assert "add_executable(Desktop2Stereo WIN32 main.cpp)" in cmake
    assert "windowscodecs" in cmake


def test_batch_prefers_native_launcher_when_present():
    source = RUN_WINDOWS.read_text(encoding="utf-8")
    assert 'if exist "%SRC_DIR%Desktop2Stereo.exe"' in source
    assert 'start "Desktop2Stereo" "%SRC_DIR%Desktop2Stereo.exe"' in source


def test_linux_launcher_has_native_x11_png_and_ready_handshake():
    source = LINUX_SOURCE.read_text(encoding="utf-8")
    cmake = LINUX_CMAKE.read_text(encoding="utf-8")
    assert "XOpenDisplay" in source
    assert "png_create_read_struct" in source
    assert "gui_ready.flag" in source
    assert "fork()" in source
    assert "pkg_check_modules(X11 REQUIRED" in cmake


def test_macos_launcher_has_appkit_splash_and_task_handshake():
    source = MACOS_SOURCE.read_text(encoding="utf-8")
    cmake = MACOS_CMAKE.read_text(encoding="utf-8")
    assert "NSWindowStyleMaskBorderless" in source
    assert "NSTask" in source
    assert "gui_ready.flag" in source
    assert "MACOSX_BUNDLE" in cmake
    assert "AppKit" in cmake


def test_remote_build_workflow_covers_all_native_launcher_platforms():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "windows-latest" in workflow
    assert "ubuntu-24.04" in workflow
    assert "macos-14" in workflow
    assert "Desktop2Stereo-linux-launcher" in workflow
    assert "Desktop2Stereo-macos-launcher" in workflow
