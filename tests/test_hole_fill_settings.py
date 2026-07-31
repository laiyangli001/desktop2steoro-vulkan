from __future__ import annotations

from gui.localization import (
    display_to_hole_fill_mode,
    hole_fill_mode_options,
    hole_fill_mode_to_display,
)
from stereo_runtime.adapter import runtime_config_from_d2s_settings, stereo_config_from_runtime


def test_hole_fill_off_option_is_localized_and_round_trips() -> None:
    assert hole_fill_mode_options("EN")[0] == "Off / No Fill"
    assert hole_fill_mode_options("ZH")[0] == "关闭 / 不补洞"
    assert hole_fill_mode_to_display("none", "ZH") == "关闭 / 不补洞"
    assert display_to_hole_fill_mode("Off / No Fill") == "none"
    assert display_to_hole_fill_mode("关闭 / 不补洞") == "none"


def test_hole_fill_off_settings_disable_runtime_fill() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Hole Fill Mode": "关闭 / 不补洞",
            "Hole Fill Radius": 3,
            "Hole Fill Strength": 1.0,
        },
        cache_dir="models",
        device="cpu",
        depth_only=True,
    )

    assert config.hole_fill == "none"
    assert config.hole_fill_mode == "none"
    assert config.hole_fill_radius == 0
    assert config.hole_fill_strength == 0.0

    stereo_config = stereo_config_from_runtime(config)
    assert stereo_config.hole_fill == "none"
    assert stereo_config.hole_fill_mode == "none"
    assert stereo_config.hole_fill_radius == 0
    assert stereo_config.hole_fill_strength == 0.0
