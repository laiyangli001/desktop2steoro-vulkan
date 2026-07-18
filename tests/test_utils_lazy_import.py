import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run_python(code, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_light_utils_import_does_not_load_settings(tmp_path):
    result = _run_python(
        "from utils import OS_NAME, read_yaml; print(bool(OS_NAME)); print(callable(read_yaml))",
        tmp_path,
    )

    assert result.stdout.splitlines() == ["True", "True"]


