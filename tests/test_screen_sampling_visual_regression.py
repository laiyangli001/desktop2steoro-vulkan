from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from stereo_runtime.screen_sampling_visual_regression import (
    compare_screen_sampling_capture_dirs,
)


def _write_eye(path, value: int) -> None:
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_screen_sampling_comparison_is_pixel_exact_and_writes_heatmaps(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    mip = tmp_path / "mip"
    for root in (legacy, mip):
        root.mkdir()
        _write_eye(root / "03_vulkan_output_left_eye.png", 10)
        _write_eye(root / "03_vulkan_output_right_eye.png", 10)
        _write_eye(root / "06_openxr_projection_left_eye.png", 10)
        _write_eye(root / "06_openxr_projection_right_eye.png", 10)
        _write_eye(root / "07_filament_screen_left_eye.png", 10)
        _write_eye(root / "07_filament_screen_right_eye.png", 10)
    changed = np.full((2, 3, 3), 12, dtype=np.uint8)
    Image.fromarray(changed, mode="RGB").save(
        mip / "07_filament_screen_left_eye.png"
    )

    result = compare_screen_sampling_capture_dirs(legacy, mip)

    assert result["pairs"]["left"]["different_pixels"] == 6
    assert result["pairs"]["left"]["max_channel_error"] == 2
    assert result["pairs"]["right"]["exact_pixels"] == 6
    assert result["source_verification"]["different_pixel_ratio"] == 0.0
    assert (mip / "screen_sampling_comparison/07_filament_screen_left_diff_heatmap.png").is_file()
    assert (mip / "screen_sampling_comparison/screen_sampling_pixel_comparison.json").is_file()


def test_screen_sampling_comparison_validates_screen_sampling_manifest(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    mip = tmp_path / "mip"
    for root in (legacy, mip):
        root.mkdir()
        _write_eye(root / "03_vulkan_output_left_eye.png", 10)
        _write_eye(root / "03_vulkan_output_right_eye.png", 10)
        _write_eye(root / "06_openxr_projection_left_eye.png", 10)
        _write_eye(root / "06_openxr_projection_right_eye.png", 10)
        _write_eye(root / "07_filament_screen_left_eye.png", 10)
        _write_eye(root / "07_filament_screen_right_eye.png", 10)
    (legacy / "screen_sampling_runtime_manifest.json").write_text(
        json.dumps(
            {
                "screen_sampling_mode": "legacy",
                "screen_sampling_update": "legacy_lod0",
            }
        ),
        encoding="utf-8",
    )
    (mip / "screen_sampling_runtime_manifest.json").write_text(
        json.dumps(
            {
                "screen_sampling_mode": "mip",
                "screen_sampling_update": "dynamic_per_frame_mip",
            }
        ),
        encoding="utf-8",
    )

    result = compare_screen_sampling_capture_dirs(legacy, mip)

    assert result["manifest_status"]["legacy"].startswith("validated")
    assert result["manifest_status"]["mip"].startswith("validated")
    assert result["manifest_paths"]["mip"].endswith(
        "screen_sampling_runtime_manifest.json"
    )


def test_screen_sampling_comparison_rejects_different_dimensions(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    mip = tmp_path / "mip"
    legacy.mkdir()
    mip.mkdir()
    _write_eye(legacy / "03_vulkan_output_left_eye.png", 10)
    _write_eye(mip / "03_vulkan_output_left_eye.png", 10)
    _write_eye(legacy / "03_vulkan_output_right_eye.png", 10)
    _write_eye(mip / "03_vulkan_output_right_eye.png", 10)
    _write_eye(legacy / "06_openxr_projection_left_eye.png", 10)
    _write_eye(legacy / "07_filament_screen_left_eye.png", 10)
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8), mode="RGB").save(
        mip / "07_filament_screen_left_eye.png"
    )
    _write_eye(legacy / "06_openxr_projection_right_eye.png", 10)
    _write_eye(mip / "06_openxr_projection_right_eye.png", 10)
    _write_eye(legacy / "07_filament_screen_right_eye.png", 10)
    _write_eye(mip / "07_filament_screen_right_eye.png", 10)

    with pytest.raises(ValueError, match="identical image shapes"):
        compare_screen_sampling_capture_dirs(legacy, mip)


def test_screen_sampling_comparison_rejects_wrong_capture_mode_manifest(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    mip = tmp_path / "mip"
    for root in (legacy, mip):
        root.mkdir()
        _write_eye(root / "03_vulkan_output_left_eye.png", 10)
        _write_eye(root / "03_vulkan_output_right_eye.png", 10)
        _write_eye(root / "06_openxr_projection_left_eye.png", 10)
        _write_eye(root / "06_openxr_projection_right_eye.png", 10)
        _write_eye(root / "07_filament_screen_left_eye.png", 10)
        _write_eye(root / "07_filament_screen_right_eye.png", 10)
    (legacy / "visual_regression_runtime_manifest.json").write_text(
        '{"screen_sampling_mode":"mip"}', encoding="utf-8"
    )
    (mip / "visual_regression_runtime_manifest.json").write_text(
        '{"screen_sampling_mode":"mip"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="expected 'legacy'"):
        compare_screen_sampling_capture_dirs(legacy, mip)
