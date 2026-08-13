import pytest

from xr_viewer.settings_menu import (
    OpenXrSettingsMenu,
    PICTURE_DEFAULTS,
    clamp_picture_values,
)


def test_picture_layout_exposes_all_planned_controls():
    menu = OpenXrSettingsMenu()
    keys = {control.key for control in menu.controls()}
    assert {
        "openxr_render_scale", "color_brightness", "color_contrast", "color_saturation", "color_gamma",
        "color_temperature", "color_tint", "vulkan_projection_min_lod",
        "vulkan_projection_max_lod", "vulkan_projection_mip_lod_bias",
        "vulkan_projection_rcas_sharpness",
    } <= keys


def test_slider_hit_and_quantization():
    menu = OpenXrSettingsMenu()
    brightness = next(control for control in menu.controls() if control.key == "color_brightness")
    x0, y0, x1, y1 = brightness.rect
    control = menu.hit_test(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
    assert control is not None and control.key == "color_brightness"
    assert control.value_from_u((x0 + x1) * 0.5) == 1.1


def test_outside_click_opens_only_after_release():
    menu = OpenXrSettingsMenu()
    assert menu.sample_trigger(0, 0.8, outside_targets=True) is False
    assert menu.sample_trigger(0, 0.2, outside_targets=True) is True
    assert menu.sample_trigger(1, 0.8, outside_targets=False) is False
    assert menu.sample_trigger(1, 0.2, outside_targets=False) is False


def test_disabled_curve_control_does_not_hit():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    subtle = next(
        control for control in menu.controls(allow_curve=False)
        if control.key == "screen:type:subtle"
    )
    x0, y0, x1, y1 = subtle.rect
    assert menu.hit_test(
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5), allow_curve=False
    ) is None


def test_min_lod_never_exceeds_max_lod():
    values = clamp_picture_values({
        "vulkan_projection_min_lod": 1.5,
        "vulkan_projection_max_lod": 0.5,
    })
    assert values["vulkan_projection_min_lod"] == 0.5


def test_tab_switch_rebuilds_page_controls():
    menu = OpenXrSettingsMenu()
    assert menu.set_tab("screen") is True
    keys = {control.key for control in menu.controls()}
    assert {
        "screen:width", "screen:height", "screen:type:flat",
        "screen:type:subtle", "screen:type:medium", "screen:type:deep",
    } <= keys
    assert "color_brightness" not in keys


def test_depth_tab_exposes_runtime_depth_controls():
    menu = OpenXrSettingsMenu()
    assert menu.set_tab("depth") is True
    keys = {control.key for control in menu.controls()}
    assert {
        "depth_strength", "depth:toggle_stereo",
        "depth:toggle_cross_eyed", "depth:reset_defaults",
    } <= keys
    depth = next(control for control in menu.controls() if control.key == "depth_strength")
    assert (depth.minimum, depth.maximum, depth.step) == (0.0, 1.0, 0.05)


def test_glow_tab_is_visible_only_for_default_environment():
    menu = OpenXrSettingsMenu()
    assert "tab:glow" not in {control.key for control in menu.controls()}
    assert "tab:glow" in {
        control.key for control in menu.controls(show_glow=True)
    }
    menu.set_tab("glow")
    assert {
        "glow:surround", "glow:glow", "glow:veil", "glow:off",
    } <= {control.key for control in menu.controls(show_glow=True)}
    assert not any(
        control.key.startswith("glow:") for control in menu.controls()
    )


def test_localized_tabs_have_compact_gaps_and_adaptive_minimum_width():
    menu = OpenXrSettingsMenu()
    english = [
        control for control in menu.controls(show_glow=True, lang="EN")
        if control.key.startswith("tab:")
    ]
    chinese = [
        control for control in menu.controls(show_glow=True, lang="CN")
        if control.key.startswith("tab:")
    ]
    assert all(control.rect[2] - control.rect[0] >= 0.135 for control in english)
    assert all(
        right.rect[0] - left.rect[2] == pytest.approx(0.008)
        for left, right in zip(english, english[1:])
    )
    assert len({round(control.rect[2] - control.rect[0], 4) for control in english}) > 1
    assert len({round(control.rect[2] - control.rect[0], 4) for control in chinese}) == 1


def test_screen_tab_exposes_distance_rotation_and_reset():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    keys = {control.key for control in menu.controls()}
    assert {
        "screen:distance", "screen:rotate:-90",
        "screen:rotate:+90", "screen:reset_defaults",
    } <= keys
    height = next(control for control in menu.controls() if control.key == "screen:height")
    assert (height.minimum, height.maximum, height.step) == (-10.0, 10.0, 0.05)


def test_room_tab_exposes_models_three_seats_and_live_sliders():
    menu = OpenXrSettingsMenu()
    menu.room_models = (("3d_a", "Room A"), ("3d_b", "Room B"))
    menu.set_tab("room")
    controls = {control.key: control for control in menu.controls()}
    assert {
        "room:model:3d_a", "room:model:3d_b",
        "room:seat:front", "room:seat:middle", "room:seat:back",
        "room:seat_height", "room:exposure",
    } <= controls.keys()
    assert (controls["room:seat_height"].minimum, controls["room:seat_height"].maximum) == (-3.0, 3.0)
    assert (controls["room:exposure"].minimum, controls["room:exposure"].maximum) == (-8.0, 8.0)
    assert [
        controls[f"room:seat:{seat}"].label
        for seat in ("front", "middle", "back")
    ] == ["Front", "Middle", "Back"]


def test_picture_layout_reserves_header_tabs_and_reset_rows():
    menu = OpenXrSettingsMenu()
    controls = {control.key: control for control in menu.controls()}
    assert controls["tab:picture"].rect[3] < controls["color_brightness"].rect[1]
    assert controls["vulkan_projection_rcas_sharpness"].rect[3] < controls["picture:reset_defaults"].rect[1]
    assert set(PICTURE_DEFAULTS) == {
        key for key, control in controls.items() if control.kind == "slider"
    }
    assert "close" not in controls


def test_openxr_render_scale_uses_half_to_double_range():
    menu = OpenXrSettingsMenu()
    control = next(
        item for item in menu.controls() if item.key == "openxr_render_scale"
    )
    assert (control.minimum, control.maximum, control.step) == (0.5, 2.0, 0.05)
    assert control.value_from_u(control.rect[0]) == 0.5
    assert control.value_from_u(control.rect[2]) == 2.0


def test_slider_minus_and_plus_are_independent_hit_targets():
    menu = OpenXrSettingsMenu()
    slider = next(
        item for item in menu.controls() if item.key == "color_brightness"
    )
    minus = next(
        item for item in menu.controls()
        if item.key == "step:minus:color_brightness"
    )
    plus = next(
        item for item in menu.controls()
        if item.key == "step:plus:color_brightness"
    )
    for expected, control in ((minus.key, minus), (plus.key, plus)):
        x0, y0, x1, y1 = control.rect
        hit = menu.hit_test(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        assert hit is not None and hit.key == expected
    assert minus.rect[2] < slider.rect[0]
    assert plus.rect[0] > slider.rect[2]
