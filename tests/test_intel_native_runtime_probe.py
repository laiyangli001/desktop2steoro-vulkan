from __future__ import annotations

from tools import intel_native_runtime_probe


def test_intel_native_runtime_probe_reports_published_bundle() -> None:
    report = intel_native_runtime_probe.build_report()
    artifacts = report["artifacts"]
    assert artifacts["all_files_present"] is True
    assert artifacts["all_hashes_match"] is True
    assert set(artifacts["files"]) == {
        "capture",
        "final_sbs_surface",
        "encoder",
        "onevpl_runtime",
        "inference",
    }


def test_intel_native_runtime_probe_is_non_strict_without_target_gpu() -> None:
    assert intel_native_runtime_probe.main([]) == 0
