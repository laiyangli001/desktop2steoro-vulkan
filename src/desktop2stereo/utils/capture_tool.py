from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_capture_select():
    # Load the selector without importing capture.__init__, which still depends
    # on utils during the package migration.
    path = Path(__file__).resolve().parents[1] / "capture" / "capture_select.py"
    spec = importlib.util.spec_from_file_location("_d2s_capture_select", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_capture_tool(setting_value: str) -> str:
    return _load_capture_select().resolve_capture_tool(setting_value)
