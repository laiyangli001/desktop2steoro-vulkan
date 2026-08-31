from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SOURCE = ROOT / "native" / "launcher" / "windows" / "main.cpp"
WINDOWS_CMAKE = ROOT / "native" / "launcher" / "windows" / "CMakeLists.txt"
RUN_WINDOWS = ROOT / "src" / "run_windows.bat"


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
