from __future__ import annotations

from pathlib import Path

from app_runtime.probe import build_capability_report
from stereo_runtime.vulkan_graph import VulkanGraphState, VulkanStereoSubmission


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_capability_report_identifies_new_project():
    report = build_capability_report()
    assert report["project"] == "desktop2steoro-vulkan"
    assert report["migration"]["python_vulkan_runtime"] == "scaffold"


def test_current_style_source_layout_is_present():
    expected = [
        "src/main.py",
        "src/app_runtime",
        "src/capture",
        "src/gui",
        "src/stereo_runtime",
        "src/viewer",
        "src/xr_viewer",
    ]
    for relative in expected:
        assert (PROJECT_ROOT / relative).exists(), relative


def test_forbidden_legacy_runtime_directories_are_not_migrated():
    forbidden = [
        "src/xr_viewer/panda_runtime",
        "src/capture/dxgi/native",
    ]
    for relative in forbidden:
        assert not (PROJECT_ROOT / relative).exists(), relative


def test_vulkan_submission_contract_is_python_native():
    submission = VulkanStereoSubmission(
        frame_id=7,
        rgb_handle=object(),
        depth_handle=object(),
        config_version=3,
    )
    assert submission.frame_id == 7
    assert VulkanGraphState.CREATED.value == "created"
