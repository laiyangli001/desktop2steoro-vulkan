from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_intel_vulkan_onevpl_smoke_is_target_only_and_strict() -> None:
    source = (
        ROOT / "src" / "tools" / "intel_vulkan_onevpl_smoke.py"
    ).read_text(encoding="utf-8")
    assert "IntelVulkanSbsRuntimeBridge" in source
    assert "OneVPLD3D11SurfaceEncoder" in source
    assert "producer_ready" in source
    assert '"zero_copy": False' in source
    assert "media_mtx_mux_checked" in source


def test_intel_vulkan_frame_contract_keeps_zero_copy_unverified() -> None:
    source = (
        ROOT
        / "src"
        / "desktop2stereo"
        / "stereo_runtime"
        / "intel_vulkan_sbs.py"
    ).read_text(encoding="utf-8")
    assert "producer_ready: bool = False" in source
    assert "ready_timeline: int = 0" in source
    assert "zero_copy: bool = False" in source
    assert "gpu_copy_count: int = 1" in source
