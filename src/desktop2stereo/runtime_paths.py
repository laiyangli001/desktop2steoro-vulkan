"""Application-local runtime paths.

This configuration is intentionally separate from the repository-level
``project_paths.env``. Values are relative to the application directory, so a
release bundle can rename that directory without changing application code.
"""

from __future__ import annotations

import os
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
_CONFIG_NAME = "runtime_paths.env"


def _read_config() -> dict[str, str]:
    override = os.environ.get("D2S_RUNTIME_PATHS_FILE")
    config = Path(override).expanduser() if override else _APP_ROOT / _CONFIG_NAME
    if not config.is_file():
        raise FileNotFoundError(f"Application runtime path config not found: {config}")
    values: dict[str, str] = {}
    for raw_line in config.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def app_root() -> Path:
    return _APP_ROOT


def python_root() -> Path:
    value = _read_config().get("D2S_PYTHON_DIR")
    if not value:
        raise ValueError(f"D2S_PYTHON_DIR is missing from {_APP_ROOT / _CONFIG_NAME}")
    return (_APP_ROOT / value).resolve()


def python_site_packages() -> Path:
    return python_root() / ("Lib" if os.name == "nt" else "lib") / (
        f"python{os.sys.version_info.major}.{os.sys.version_info.minor}/site-packages"
        if os.name != "nt"
        else "site-packages"
    )
