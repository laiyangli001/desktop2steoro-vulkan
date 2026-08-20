"""Load the repository/package paths from project_paths.env.

The environment file is the single layout source for tests and developer tools.
Application code should resolve assets from its own package directory instead of
hard-coding the repository's src/ layout.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


_CONFIG_NAME = "project_paths.env"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def find_config(start: Path | None = None) -> Path:
    override = os.environ.get("D2S_PATHS_FILE")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"D2S_PATHS_FILE does not exist: {candidate}")
        return candidate

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / _CONFIG_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to locate {_CONFIG_NAME} from {current}")


@dataclass(frozen=True)
class ProjectPaths:
    config_file: Path
    repo_root: Path
    app_dir: Path
    python_dir: Path
    env_install_dir: Path

    @property
    def python_executable(self) -> Path:
        name = "python.exe" if os.name == "nt" else "bin/python"
        return self.python_dir / name


def load_project_paths(start: Path | None = None) -> ProjectPaths:
    config_file = find_config(start)
    values = _read_env_file(config_file)
    repo_root = config_file.parent

    def resolve(name: str) -> Path:
        value = values.get(name)
        if not value:
            raise ValueError(f"Missing {name} in {config_file}")
        return (repo_root / value).resolve()

    return ProjectPaths(
        config_file=config_file,
        repo_root=repo_root,
        app_dir=resolve("D2S_APP_DIR"),
        python_dir=resolve("D2S_PYTHON_DIR"),
        env_install_dir=resolve("D2S_ENV_INSTALL_DIR"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a configured Desktop2Stereo path")
    parser.add_argument("key", choices=("repo", "app", "python", "env_install", "config"))
    args = parser.parse_args()
    paths = load_project_paths()
    selected = {
        "repo": paths.repo_root,
        "app": paths.app_dir,
        "python": paths.python_dir,
        "env_install": paths.env_install_dir,
        "config": paths.config_file,
    }[args.key]
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
