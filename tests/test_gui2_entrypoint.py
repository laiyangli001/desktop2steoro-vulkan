from pathlib import Path


ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "src" / "desktop2stereo" / "app_runtime" / "bootstrap.py"
LEGACY_GUI = ROOT / "src" / "desktop2stereo" / "gui" / "gui.py"
GUI2 = ROOT / "src" / "desktop2stereo" / "gui2" / "gui.py"

import sys

sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))

from app_runtime.gui_selection import (  # noqa: E402
    LEGACY_GUI as LEGACY_GUI_ID,
    MODERN_GUI as MODERN_GUI_ID,
    read_startup_gui,
    save_startup_gui,
)


def test_gui2_is_default_and_explicit_gui_flags_remain_supported() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'parser.add_argument("--gui2"' in source
    assert "from gui2.gui import main as gui2_main" in source
    assert "from gui.gui import main as gui_main" in source
    assert "read_startup_gui()" in source
    assert "MODERN_GUI" in source
    assert "LEGACY_GUI" in source


def test_gui2_shell_is_separate_from_legacy_gui_file() -> None:
    source = GUI2.read_text(encoding="utf-8")
    assert "class Desktop2StereoGUI2" in source
    assert "from gui.gui import Desktop2StereoGUI" in source
    assert LEGACY_GUI.exists()
    assert not (ROOT / "src" / "desktop2stereo" / "gui2" / "flet_packages").exists()


def test_app_runtime_does_not_eagerly_import_processing_runtime() -> None:
    source = (ROOT / "src" / "desktop2stereo" / "app_runtime" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "from .runtime_entry import run_processing_runtime\n" not in source
    assert "def run_processing_runtime" in source


def test_startup_gui_preference_defaults_to_gui2_and_preserves_settings(tmp_path) -> None:
    settings_path = tmp_path / "settings.yaml"
    assert read_startup_gui(settings_path) == MODERN_GUI_ID
    settings_path.write_text("Language: CN\n", encoding="utf-8")

    ok, error = save_startup_gui(LEGACY_GUI_ID, settings_path)
    assert ok, error
    assert read_startup_gui(settings_path) == LEGACY_GUI_ID
    assert "Language: CN" in settings_path.read_text(encoding="utf-8")

    ok, error = save_startup_gui(MODERN_GUI_ID, settings_path)
    assert ok, error
    assert read_startup_gui(settings_path) == MODERN_GUI_ID
