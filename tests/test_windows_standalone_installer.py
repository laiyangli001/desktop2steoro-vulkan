from pathlib import Path

from path_config import ENV_INSTALL_ROOT

REPO_ROOT = ENV_INSTALL_ROOT.parents[1]


def test_cuda_installer_uses_complete_project_local_python() -> None:
    script = (ENV_INSTALL_ROOT / "install-cuda_standalone.bat").read_text(
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
    assert 'Set "PYTHON_ROOT=%PROJECT_ROOT%\\python3"' in script
    assert 'Set "PIP_RETRIES=10"' in script
    assert 'Set "PIP_TIMEOUT=180"' in script
    assert script.count("call :install_cuda_requirements") == 3


def test_rocm_installer_uses_and_validates_project_local_python() -> None:
    script = (ENV_INSTALL_ROOT / "install-rocm7-standalone.bat").read_text(
        encoding="utf-8"
    )

    assert 'set "PYTHON_ROOT=%PROJECT_ROOT%\\python3"' in script
    assert 'if not exist "%PYTHON_EXE%"' in script
    assert "sys.version_info[:2] == (3, 12)" in script
    assert '"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%"' in script
    assert "Failed to install ROCm requirements" in script
    assert "Failed to initialize the ROCm SDK" in script


def test_posix_installers_use_the_virtualenv_python_and_current_layout() -> None:
    for name in ("install-cuda.bash", "install-mps", "install-mps0", "install-rocm7.bash"):
        script = (ENV_INSTALL_ROOT / name).read_text(encoding="utf-8")
        assert 'PYTHON_EXE="$VIRTUAL_ENV/bin/python"' in script
        assert "export PIP_RETRIES=10" in script
        assert "export PIP_TIMEOUT=180" in script

    for name in ("install-mps", "install-mps0"):
        script = (ENV_INSTALL_ROOT / name).read_text(encoding="utf-8")
        assert 'chmod a+x "$PROJECT_ROOT/run_mac" "$0"' in script
        assert '"$PROJECT_ROOT/../run_mac"' not in script


def test_windows_scripts_are_preserved_as_crlf_bytes() -> None:
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat -text" in attributes
    assert "*.cmd -text" in attributes
    assert "*.reg -text" in attributes

    windows_files = [
        path
        for pattern in ("*.bat", "*.cmd", "*.reg")
        for path in REPO_ROOT.rglob(pattern)
        if ".git" not in path.parts and "..python3" not in path.parts
    ]
    assert windows_files
    for path in windows_files:
        data = path.read_bytes()
        assert b"\n" not in data.replace(b"\r\n", b""), path
