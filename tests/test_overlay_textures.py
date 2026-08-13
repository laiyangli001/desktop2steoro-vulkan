from __future__ import annotations

import numpy as np

from gui.localization import MESSAGE_CATALOGS, gettext_for, normalize_locale
from xr_viewer.overlay_textures import (
    build_settings_menu_rgba,
    build_screen_adjust_osd_rgba,
    build_screen_preset_osd_rgba,
)
from xr_viewer.settings_menu import OpenXrSettingsMenu, SETTINGS_MENU_TEXTURE_SIZE


def test_openxr_settings_menu_text_uses_every_gui_locale_catalog():
    keys = (
        "Picture", "Depth", "Glow", "Room", "Screen",
        "Surround Glow", "Veil", "OFF", "Glow effects",
    )
    for locale, catalog in MESSAGE_CATALOGS.items():
        assert all(key in catalog for key in keys), locale
        assert all(gettext_for(locale, key) == catalog[key] for key in keys)
    assert normalize_locale("zh-CN") == "CN"


def test_settings_menu_canvas_has_room_below_bottom_controls():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    rgba = build_settings_menu_rgba(menu, {}, lang="CN")
    assert rgba.shape[:2] == (
        SETTINGS_MENU_TEXTURE_SIZE[1], SETTINGS_MENU_TEXTURE_SIZE[0]
    )
    reset = next(
        control for control in menu.controls()
        if control.key == "screen:reset_defaults"
    )
    assert int(reset.rect[3] * rgba.shape[0]) < rgba.shape[0] - 18


def test_screen_preset_osd_matches_legacy_colors_and_centering():
    rgba = build_screen_preset_osd_rgba("1000\" IMAX")

    assert rgba.shape == (78, 768, 4)
    assert np.any(np.all(rgba[:, :, :3] == (32, 32, 36), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (150, 158, 185), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (0, 210, 230), axis=2))


def test_screen_adjust_osd_matches_legacy_centered_style():
    rgba = build_screen_adjust_osd_rgba(2.4, 3.5)

    assert rgba.shape == (78, 512, 4)
    assert rgba.dtype == np.uint8
    assert rgba.flags.c_contiguous
    assert tuple(rgba[0, 0]) == (0, 0, 0, 0)
    assert np.any(np.all(rgba[:, :, :3] == (32, 32, 36), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (150, 158, 185), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (0, 210, 230), axis=2))

    alpha = rgba[:, :, 3]
    occupied = np.where(alpha > 0)
    assert occupied[1].min() < 32
    assert occupied[1].max() > 480
