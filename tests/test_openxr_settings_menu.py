from xr_viewer.settings_menu import OpenXrSettingsMenu, clamp_picture_values


def test_picture_layout_exposes_all_planned_controls():
    menu = OpenXrSettingsMenu()
    keys = {control.key for control in menu.controls()}
    assert {
        "color_brightness", "color_contrast", "color_saturation", "color_gamma",
        "color_temperature", "color_tint", "vulkan_projection_min_lod",
        "vulkan_projection_max_lod", "vulkan_projection_mip_lod_bias",
        "vulkan_projection_rcas_sharpness",
    } <= keys


def test_slider_hit_and_quantization():
    menu = OpenXrSettingsMenu()
    control = menu.hit_test((0.08 + 0.38 * 0.5, 0.19 + 0.08))
    assert control is not None and control.key == "color_brightness"
    assert control.value_from_u(0.08 + 0.38 * 0.5) == 1.1


def test_outside_click_opens_only_after_release():
    menu = OpenXrSettingsMenu()
    assert menu.sample_trigger(0, 0.8, outside_targets=True) is False
    assert menu.sample_trigger(0, 0.2, outside_targets=True) is True
    assert menu.sample_trigger(1, 0.8, outside_targets=False) is False
    assert menu.sample_trigger(1, 0.2, outside_targets=False) is False


def test_disabled_curve_control_does_not_hit():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    assert menu.hit_test((0.5, 0.75), allow_curve=False) is None


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
    assert {"screen:width", "screen:height", "screen:curve"} <= keys
    assert "color_brightness" not in keys
