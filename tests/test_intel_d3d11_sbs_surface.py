from __future__ import annotations

import numpy as np

from desktop2stereo.stereo_runtime.providers.intel.d3d11_sbs_surface import (
    D3D11SbsSurface,
    probe_d3d11_sbs_surface,
)


def test_d3d11_sbs_surface_probe_is_safe_without_optional_dll() -> None:
    result = probe_d3d11_sbs_surface()
    assert result["backend"] == "d3d11_sbs_surface"
    assert isinstance(result["available"], bool)
    assert result["zero_copy"] is False
    assert isinstance(result["external_bgra_texture_import"], bool)
    assert isinstance(result["external_texture_zero_copy_ready"], bool)


def test_rgb_to_bgra_preserves_final_sbs_geometry_and_channels() -> None:
    from streaming.direct_sbs import IntelD3D11DirectSbsOutput

    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    bgra = IntelD3D11DirectSbsOutput._rgb_to_bgra(rgb)
    assert bgra.shape == (1, 2, 4)
    assert bgra.tolist() == [[[3, 2, 1, 255], [6, 5, 4, 255]]]


def test_d3d11_surface_requires_optional_bridge() -> None:
    result = probe_d3d11_sbs_surface()
    if result["available"]:
        return
    try:
        D3D11SbsSurface(width=64, height=64)
    except RuntimeError as exc:
        assert "d3d11" in str(exc).lower()
    else:
        raise AssertionError("D3D11 SBS surface unexpectedly initialized without bridge")
