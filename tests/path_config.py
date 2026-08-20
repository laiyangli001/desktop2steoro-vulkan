"""Shared test paths loaded from the repository layout configuration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import load_project_paths  # noqa: E402


PATHS = load_project_paths(PROJECT_ROOT)
APP_ROOT = PATHS.app_dir
PYTHON_ROOT = PATHS.python_dir
ENV_INSTALL_ROOT = PATHS.env_install_dir
