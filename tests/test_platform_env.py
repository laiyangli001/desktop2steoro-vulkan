from pathlib import Path

from utils import platform_env


_ROCM_ENV_KEYS = [
    "MIOPEN_ENABLE_LOGGING",
    "HIP_PLATFORM",
    "HIP_PATH",
    "HIP_CLANG_PATH",
    "HIP_INCLUDE_PATH",
    "HIP_LIB_PATH",
    "HIP_DEVICE_LIB_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "PKG_CONFIG_PATH",
]


def _make_rocm_sdk(root: Path) -> Path:
    rocm = root / "_rocm_sdk_devel"
    for relative in (
        "bin",
        "include",
        "lib",
        "lib64",
        "llvm/bin",
        "lib/llvm/amdgcn/bitcode",
        "lib/pkgconfig",
    ):
        (rocm / relative).mkdir(parents=True, exist_ok=True)
    return rocm


def test_configure_rocm_environment_sets_windows_hip_paths(monkeypatch, tmp_path):
    rocm = _make_rocm_sdk(tmp_path)
    for key in _ROCM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", "C:\\Windows")

    resolved = platform_env.configure_rocm_environment("Windows", rocm)

    assert resolved == str(rocm.resolve())
    assert platform_env.os.environ["MIOPEN_ENABLE_LOGGING"] == "0"
    assert platform_env.os.environ["HIP_PLATFORM"] == "amd"
    assert platform_env.os.environ["HIP_PATH"] == str(rocm.resolve())
    assert platform_env.os.environ["HIP_CLANG_PATH"] == str(rocm.resolve() / "llvm" / "bin")
    assert platform_env.os.environ["HIP_INCLUDE_PATH"] == str(rocm.resolve() / "include")
    assert platform_env.os.environ["HIP_LIB_PATH"] == str(rocm.resolve() / "lib")
    assert platform_env.os.environ["HIP_DEVICE_LIB_PATH"] == str(rocm.resolve() / "lib" / "llvm" / "amdgcn" / "bitcode")
    path_parts = platform_env.os.environ["PATH"].split(platform_env.os.pathsep)
    assert path_parts[:2] == [str(rocm.resolve() / "bin"), str(rocm.resolve() / "llvm" / "bin")]


def test_configure_rocm_environment_ignores_non_windows(monkeypatch, tmp_path):
    rocm = _make_rocm_sdk(tmp_path)
    monkeypatch.delenv("HIP_PATH", raising=False)

    resolved = platform_env.configure_rocm_environment("Darwin", rocm)

    assert resolved is None
    assert "HIP_PATH" not in platform_env.os.environ


def test_configure_rocm_environment_does_not_duplicate_existing_path(monkeypatch, tmp_path):
    rocm = _make_rocm_sdk(tmp_path).resolve()
    for key in _ROCM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", str(rocm / "bin"))

    platform_env.configure_rocm_environment("Windows", rocm)
    platform_env.configure_rocm_environment("Windows", rocm)

    path_parts = platform_env.os.environ["PATH"].split(platform_env.os.pathsep)
    assert path_parts.count(str(rocm / "bin")) == 1
    assert path_parts.count(str(rocm / "llvm" / "bin")) == 1
