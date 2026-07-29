from __future__ import annotations

from stereo_runtime.adapter import runtime_config_from_d2s_settings


def test_screen_sampling_visual_regression_settings_are_program_controlled() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "OpenXR Visual Regression Directory": "artifacts/screen-mip",
        },
        cache_dir="models",
        device="cpu",
        depth_only=True,
    )

    assert config.openxr_visual_regression_dir == "artifacts/screen-mip"
