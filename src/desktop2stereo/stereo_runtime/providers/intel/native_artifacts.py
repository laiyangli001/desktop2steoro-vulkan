"""Shared discovery for the optional Intel Windows native artifact bundle.

GitHub Actions produces the four Intel bridge DLLs as one artifact.  Keeping
the directory discovery here lets a downloaded bundle be used by the capture,
inference, surface, and encoder adapters without four separate environment
variables.  Individual DLL environment variables remain supported as an
explicit override.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def native_artifact_directories() -> list[Path]:
    """Return ordered directories that may contain the Intel bridge bundle."""
    directories: list[Path] = []
    for variable in ("D2S_INTEL_NATIVE_ARTIFACT_DIR", "D2S_INTEL_NATIVE_DIR"):
        configured = os.environ.get(variable, "").strip()
        if configured:
            directories.extend(Path(item) for item in configured.split(os.pathsep) if item)

    root = _repository_root()
    directories.extend(
        [
            root / "native" / "intel-windows-artifact",
            root / "src" / "desktop2stereo" / "native" / "windows",
            root / "native" / "windows",
        ]
    )

    result: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        key = os.path.normcase(os.path.abspath(os.fspath(directory)))
        if key not in seen:
            seen.add(key)
            result.append(directory)
    return result


def native_dll_candidates(
    filename: str,
    *,
    environment_variable: str,
    extra_directories: tuple[Path, ...] = (),
) -> list[Path]:
    """Return an explicit override followed by shared and legacy locations."""
    candidates: list[Path] = []
    configured = os.environ.get(environment_variable, "").strip()
    if configured:
        candidates.append(Path(configured))

    directories = [*extra_directories, *native_artifact_directories()]
    candidates.extend(directory / filename for directory in directories)

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


__all__ = ["native_artifact_directories", "native_dll_candidates"]
