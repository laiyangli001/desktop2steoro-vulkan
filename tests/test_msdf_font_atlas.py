from pathlib import Path
import json

import numpy as np

from xr_viewer.msdf_font_atlas import MsdfFontAtlas


def test_bundled_msdf_atlas_has_all_pages_and_glyphs():
    atlas = MsdfFontAtlas()
    assert 3 <= len(atlas.pages) <= 4
    assert len(atlas.glyphs) >= 3500
    assert atlas.page_width > 0
    assert atlas.page_height > 0
    assert all(atlas.page_path(index).is_file() for index in range(len(atlas.pages)))


def test_bundled_msdf_atlas_uses_charset_ordered_grid():
    atlas = MsdfFontAtlas()
    metadata = json.loads(
        (atlas.root / "d2s_overlay_msdf.json").read_text(encoding="utf-8")
    )
    for index, glyph in enumerate(metadata["chars"]):
        slot = index % 1024
        assert glyph["page"] == index // 1024
        assert int(glyph["x"]) // 64 == slot % 32
        assert int(glyph["y"]) // 64 == slot // 32


def test_msdf_layout_keeps_page_and_uv_metadata():
    atlas = MsdfFontAtlas()
    instances = atlas.layout("性能 FPS")
    assert len(instances) == len("性能 FPS")
    assert all(0 <= item.page < len(atlas.pages) for item in instances)
    assert all(0.0 <= value <= 1.0 for item in instances for value in (*item.uv_min, *item.uv_max))


def test_msdf_page_and_geometry_match_native_buffer_contract():
    atlas = MsdfFontAtlas()
    page = atlas.page_rgba(0)
    assert page.ndim == 3
    assert page.shape[2] == 4
    geometry = atlas.build_geometry(
        "性能",
        transform=np.eye(4, dtype=np.float32),
        pixel_scale=(0.001, 0.001),
    )
    vertices, indices = geometry[next(iter(geometry))]
    assert vertices.dtype == np.float32
    assert vertices.shape[1] == 9
    assert vertices.flags.c_contiguous
    assert indices.dtype == np.uint16
    assert indices.flags.c_contiguous
