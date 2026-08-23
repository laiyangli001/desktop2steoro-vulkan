from __future__ import annotations

from pathlib import Path

from desktop2stereo.stereo_runtime.providers.intel.native_artifacts import (
    native_artifact_directories,
    native_dll_candidates,
)


def test_shared_artifact_directory_is_used_for_all_bridge_candidates(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "intel-bundle"
    monkeypatch.setenv("D2S_INTEL_NATIVE_ARTIFACT_DIR", str(bundle))
    monkeypatch.delenv("D2S_INTEL_NATIVE_DIR", raising=False)
    monkeypatch.delenv("D2S_OPENVINO_D3D11_DLL", raising=False)

    directories = native_artifact_directories()
    assert directories[0] == bundle

    candidates = native_dll_candidates(
        "d2s_openvino_d3d11_bridge.dll",
        environment_variable="D2S_OPENVINO_D3D11_DLL",
    )
    assert candidates[0] == bundle / "d2s_openvino_d3d11_bridge.dll"


def test_explicit_bridge_path_precedes_shared_artifact_directory(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "custom" / "bridge.dll"
    bundle = tmp_path / "intel-bundle"
    monkeypatch.setenv("D2S_OPENVINO_D3D11_DLL", str(explicit))
    monkeypatch.setenv("D2S_INTEL_NATIVE_ARTIFACT_DIR", str(bundle))

    candidates = native_dll_candidates(
        "d2s_openvino_d3d11_bridge.dll",
        environment_variable="D2S_OPENVINO_D3D11_DLL",
    )
    assert candidates[0] == explicit
    assert candidates[1] == bundle / "d2s_openvino_d3d11_bridge.dll"


def test_duplicate_native_directories_are_removed(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "intel-bundle"
    monkeypatch.setenv("D2S_INTEL_NATIVE_ARTIFACT_DIR", str(bundle))
    monkeypatch.setenv("D2S_INTEL_NATIVE_DIR", str(bundle))

    directories = native_artifact_directories()
    assert directories.count(bundle) == 1
