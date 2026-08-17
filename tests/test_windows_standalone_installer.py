from pathlib import Path


def test_cuda_installer_uses_complete_project_local_python() -> None:
    script = (Path(__file__).resolve().parents[1] / "src" / "install-cuda_standalone.bat").read_text(
        encoding="utf-8"
    )

    assert "www.nuget.org/api/v2/package/python/%PYTHON_VERSION%" in script
    assert "Expand-Archive" in script
    assert "Join-Path $staging 'tools'" in script
    assert "importlib.util.find_spec('ensurepip')" in script
    assert 'else 1)\" >nul 2>nul' in script
    assert "import ensurepip, ssl, sqlite3, sys" in script
    assert "-m ensurepip --upgrade" in script
    assert "embed-amd64.zip" not in script
    assert "get-pip.py" not in script
    assert "TargetDir=" not in script
    assert "python-%PYTHON_VERSION%-amd64.exe" not in script
