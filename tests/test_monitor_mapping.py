import ast
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = ROOT / "src" / "gui" / "builders.py"


def _load_monitor_methods(monitors, primary_index):
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GUIBuilderMixin")
    wanted = {"populate_monitors", "update_stereo_monitor_menu"}
    selected = [node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=[ast.ClassDef(name="MonitorMixin", bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "list_monitors": lambda: monitors,
        "get_primary_monitor_index": lambda: primary_index,
        "PRIMARY_MONITOR_SUFFIX": " (Primary)",
    }
    exec(compile(module, str(BUILDERS_SOURCE), "exec"), namespace)
    return namespace["MonitorMixin"]


class FakeDropdown:
    def __init__(self, value=""):
        self.options = []
        self.value = value
        self.update_count = 0

    def update(self):
        self.update_count += 1


class FakeMonitorGui:
    def __init__(self, monitor_value="", stereo_value="Viewer Window"):
        self.monitor_dd = FakeDropdown(monitor_value)
        self.stereo_monitor_dd = FakeDropdown(stereo_value)
        self.monitor_label_to_index = {}
        self.capture_mode_key = "Monitor"
        self.fit_count = 0

    def _fit_window_to_content(self):
        self.fit_count += 1


FAKE_MSS_MONITORS = [
    {"left": 0, "top": 0, "width": 9600, "height": 2160},
    {"left": 5760, "top": 0, "width": 3840, "height": 2160},
    {"left": 3840, "top": 0, "width": 1920, "height": 1080},
    {"left": 0, "top": 0, "width": 3840, "height": 2160},
]


class FakeMssContext:
    monitors = FAKE_MSS_MONITORS

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMssModule:
    @staticmethod
    def mss():
        return FakeMssContext()


class FakeWin32Api:
    @staticmethod
    def EnumDisplayMonitors():
        return [
            ("h5", None, (5760, 0, 9600, 2160)),
            ("h2", None, (3840, 0, 5760, 1080)),
            ("h1", None, (0, 0, 3840, 2160)),
        ]

    @staticmethod
    def GetMonitorInfo(handle):
        return {
            "h5": {"Device": r"\\.\DISPLAY5"},
            "h2": {"Device": r"\\.\DISPLAY2"},
            "h1": {"Device": r"\\.\DISPLAY1"},
        }[handle]


def test_list_monitors_uses_windows_display_numbers(monkeypatch):
    from gui import capture_sources

    monkeypatch.setattr(capture_sources, "OS_NAME", "Windows")
    monkeypatch.setitem(sys.modules, "mss", FakeMssModule)
    monkeypatch.setitem(sys.modules, "win32api", FakeWin32Api)

    monitors = capture_sources.list_monitors()

    assert [mon["display_number"] for mon in monitors] == [1, 2, 3]
    assert [mon["device_display_number"] for mon in monitors] == [1, 2, 5]
    assert [mon["capture_index"] for mon in monitors] == [3, 2, 1]


def test_populate_monitors_preserves_current_selection_and_updates_stereo_menu():
    monitors = [
        {"capture_index": 1, "display_number": 1, "left": 0, "top": 0, "width": 1920, "height": 1080},
        {"capture_index": 2, "display_number": 2, "left": 1920, "top": 0, "width": 2560, "height": 1440},
    ]
    current = "2: 2560x1440 @ (1920,0)"
    gui = FakeMonitorGui(monitor_value=current)
    monitor_mixin = _load_monitor_methods(monitors, primary_index=1)
    gui.update_stereo_monitor_menu = monitor_mixin.update_stereo_monitor_menu.__get__(gui, FakeMonitorGui)

    monitor_mixin.populate_monitors(gui)

    assert gui.monitor_dd.value == current
    assert gui.monitor_label_to_index[current] == 2
    assert gui.stereo_monitor_dd.options == ["Viewer Window", "1: 1920x1080 @ (0,0) (Primary)"]
    assert gui.fit_count == 1


def test_populate_monitors_falls_back_to_primary_when_current_missing():
    monitors = [
        {"capture_index": 1, "display_number": 1, "left": 0, "top": 0, "width": 1920, "height": 1080},
        {"capture_index": 2, "display_number": 2, "left": 1920, "top": 0, "width": 2560, "height": 1440},
    ]
    gui = FakeMonitorGui(monitor_value="9: stale")
    monitor_mixin = _load_monitor_methods(monitors, primary_index=2)
    gui.update_stereo_monitor_menu = monitor_mixin.update_stereo_monitor_menu.__get__(gui, FakeMonitorGui)

    monitor_mixin.populate_monitors(gui)

    primary = "2: 2560x1440 @ (1920,0) (Primary)"
    assert gui.monitor_dd.value == primary
    assert gui.stereo_monitor_dd.options == ["Viewer Window", "1: 1920x1080 @ (0,0)"]
