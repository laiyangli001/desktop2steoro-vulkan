from __future__ import annotations

from tools import intel_native_runtime_probe


def test_intel_native_runtime_probe_reports_published_bundle() -> None:
    report = intel_native_runtime_probe.build_report()
    artifacts = report["artifacts"]
    assert artifacts["all_files_present"] is True
    assert artifacts["all_hashes_match"] is True
    core_files = {
        "capture",
        "final_sbs_surface",
        "encoder",
        "onevpl_runtime",
        "inference",
    }
    assert core_files <= set(artifacts["files"])
    runtime_names = set(artifacts["manifest"]["openvino_runtime_files"])
    assert {
        f"openvino_runtime:{name}" for name in runtime_names
    } <= set(artifacts["files"])


def test_intel_native_runtime_probe_is_non_strict_without_target_gpu() -> None:
    assert intel_native_runtime_probe.main([]) == 0
