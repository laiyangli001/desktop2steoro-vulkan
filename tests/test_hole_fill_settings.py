from __future__ import annotations

from types import SimpleNamespace

from gui.localization import (
    display_to_hole_fill_mode,
    get_messages,
    hole_fill_mode_options,
    hole_fill_mode_to_display,
)
from gui.config_mgr import GUIConfigMixin
from stereo_runtime.adapter import runtime_config_from_d2s_settings, stereo_config_from_runtime


def test_hole_fill_off_option_is_localized_and_round_trips() -> None:
    assert hole_fill_mode_options("EN")[0] == "Off / No Fill"
    assert hole_fill_mode_options("ZH")[0] == "关闭 / 不补洞"
    assert hole_fill_mode_to_display("none", "ZH") == "关闭 / 不补洞"
    assert display_to_hole_fill_mode("Off / No Fill") == "none"
    assert display_to_hole_fill_mode("关闭 / 不补洞") == "none"


def test_hole_fill_options_only_expose_off_balanced_and_quality() -> None:
    assert hole_fill_mode_options("EN") == [
        "Off / No Fill",
        "Balanced / Standard",
        "Content Aware / Highest Quality",
    ]

    tooltip = get_messages("ZH")["tooltip_hole_fill_mode"]
    assert "电影模式 = 关闭 / 不补洞" in tooltip
    assert "游戏模式 = 关闭 / 不补洞" in tooltip
    assert "图片模式 = 增强 / 高质量" in tooltip


def test_gui_stereo_presets_select_expected_hole_fill_modes() -> None:
    cinema = GUIConfigMixin._stereo_preset_gui_values("cinema")
    game = GUIConfigMixin._stereo_preset_gui_values("game_low_latency")
    image = GUIConfigMixin._stereo_preset_gui_values("still_image_hq")

    assert (cinema["hole_fill_mode"], cinema["hole_fill_radius"], cinema["hole_fill_strength"]) == (
        "none",
        0,
        0.0,
    )
    assert (game["hole_fill_mode"], game["hole_fill_radius"], game["hole_fill_strength"]) == (
        "none",
        0,
        0.0,
    )
    assert (image["hole_fill_mode"], image["hole_fill_radius"], image["hole_fill_strength"]) == (
        "quality",
        3,
        1.0,
    )
    assert hole_fill_mode_options("ZH") == [
        "关闭 / 不补洞",
        "均衡 / 标准",
        "增强 / 高质量",
    ]


def test_hole_fill_mode_controls_gui_temporal_strength() -> None:
    class Host(GUIConfigMixin):
        def __init__(self):
            self.hole_fill_mode_dd = SimpleNamespace(value="关闭 / 不补洞")
            self.temporal_strength_dd = SimpleNamespace(value="0.25")
            self.stereo_preset_dd = SimpleNamespace(value="cinema")
            self.hot_save_count = 0

        def _display_to_preset(self, value):
            return value

        def on_stereo_hot_param_change(self, e=None):
            self.hot_save_count += 1

    host = Host()
    host.on_hole_fill_mode_change()

    assert host.temporal_strength_dd.value == "0.00"
    assert host.hot_save_count == 1

    host.hole_fill_mode_dd.value = "均衡 / 标准"
    host.on_hole_fill_mode_change()

    assert host.temporal_strength_dd.value == "0.00"
    assert host.hot_save_count == 2


def test_hole_fill_off_settings_disable_runtime_fill() -> None:
    config = runtime_config_from_d2s_settings(
        {
            "Depth Model": "Distill-Any-Depth-Base",
            "Hole Fill Mode": "关闭 / 不补洞",
            "Hole Fill Radius": 3,
            "Hole Fill Strength": 1.0,
            "Temporal": True,
            "Temporal Strength": 0.25,
        },
        cache_dir="models",
        device="cpu",
        depth_only=True,
    )

    assert config.hole_fill == "none"
    assert config.hole_fill_mode == "none"
    assert config.hole_fill_radius == 0
    assert config.hole_fill_strength == 0.0
    assert config.temporal is False
    assert config.temporal_strength == 0.0

    stereo_config = stereo_config_from_runtime(config)
    assert stereo_config.hole_fill == "none"
    assert stereo_config.hole_fill_mode == "none"
    assert stereo_config.hole_fill_radius == 0
    assert stereo_config.hole_fill_strength == 0.0
    assert stereo_config.temporal is False
    assert stereo_config.temporal_strength == 0.0
