"""Lightweight startup-GUI preference shared by the launcher and both GUIs."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


STARTUP_GUI_KEY = "Startup GUI"
LEGACY_GUI = "legacy"
MODERN_GUI = "gui2"
DEFAULT_STARTUP_GUI = MODERN_GUI
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "settings.yaml"


def normalize_startup_gui(value) -> str:
    text = str(value or "").strip().casefold().replace("_", "-")
    if text in {"legacy", "gui", "gui1", "old", "classic"}:
        return LEGACY_GUI
    if text in {"gui2", "new", "modern"}:
        return MODERN_GUI
    return DEFAULT_STARTUP_GUI


def read_startup_gui(settings_path: str | os.PathLike | None = None) -> str:
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    try:
        with path.open("r", encoding="utf-8-sig") as settings_file:
            settings = yaml.safe_load(settings_file) or {}
        if not isinstance(settings, dict):
            return DEFAULT_STARTUP_GUI
        return normalize_startup_gui(settings.get(STARTUP_GUI_KEY))
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return DEFAULT_STARTUP_GUI


def save_startup_gui(
    value,
    settings_path: str | os.PathLike | None = None,
) -> tuple[bool, str]:
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    try:
        settings = {}
        if path.is_file():
            with path.open("r", encoding="utf-8-sig") as settings_file:
                loaded = yaml.safe_load(settings_file) or {}
            if isinstance(loaded, dict):
                settings.update(loaded)
        settings[STARTUP_GUI_KEY] = normalize_startup_gui(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".gui.tmp")
        with temporary_path.open("w", encoding="utf-8") as settings_file:
            yaml.safe_dump(settings, settings_file, allow_unicode=True, sort_keys=False)
        os.replace(temporary_path, path)
        return True, ""
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        return False, str(exc)
