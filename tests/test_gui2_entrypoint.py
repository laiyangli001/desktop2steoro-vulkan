from pathlib import Path


ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "src" / "desktop2stereo" / "app_runtime" / "bootstrap.py"
LEGACY_GUI = ROOT / "src" / "desktop2stereo" / "gui" / "gui.py"
GUI2 = ROOT / "src" / "desktop2stereo" / "gui2" / "gui.py"


def test_gui2_has_explicit_entrypoint_and_legacy_default() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'parser.add_argument("--gui2"' in source
    assert "from gui2.gui import main as gui2_main" in source
    assert "from gui.gui import main as gui_main" in source
    assert "if args.gui2:" in source


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
