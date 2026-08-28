import json

import stereo_runtime.nvfruc as nvfruc
from stereo_runtime.nvfruc_calibration import (
    NvFrucCalibrationCache,
    NvFrucCalibrationResult,
    calibration_fingerprint,
    calculate_safe_output_limit,
    downgrade_target_fps,
    output_base_fps,
)


def test_output_base_fps_uses_final_target_semantics():
    assert output_base_fps(60) == 30
    assert output_base_fps(45) == 23
    assert output_base_fps(60, enabled=False) == 60


def test_safe_limit_uses_p95_bottleneck_and_headroom():
    maximum, safe, bottleneck = calculate_safe_output_limit(
        {"inference": [20.0, 20.0, 22.0], "nvfruc": [4.0, 5.0, 6.0]},
        output_cap_fps=90,
    )
    assert bottleneck == "inference"
    assert maximum < 50.0
    assert safe == int(maximum * 0.8)


def test_calibration_cache_is_fingerprint_scoped(tmp_path):
    fingerprint = calibration_fingerprint({"gpu": "test", "width": 3840})
    result = NvFrucCalibrationResult(60.0, 48, 30, "inference", True, 18.0)
    cache = NvFrucCalibrationCache(tmp_path)
    cache.save(fingerprint, result)
    assert cache.load(fingerprint) == result
    assert cache.load("different") is None
    payload = json.loads((tmp_path / "nvfruc-calibration.json").read_text(encoding="utf-8"))
    assert payload["fingerprint"] == fingerprint


def test_probe_reports_platform_unavailability(monkeypatch):
    monkeypatch.setattr(nvfruc.platform, "system", lambda: "Linux")
    result = nvfruc.probe_nvfruc()
    assert result.available is False
    assert "Windows" in result.reason


def test_runtime_downgrade_never_returns_zero():
    assert downgrade_target_fps(60) == 54
    assert downgrade_target_fps(1) == 1
