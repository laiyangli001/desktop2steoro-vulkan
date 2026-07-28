from __future__ import annotations

import numpy as np

from xr_viewer.overlay_textures import (
    build_screen_adjust_osd_rgba,
    build_screen_preset_osd_rgba,
)


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
