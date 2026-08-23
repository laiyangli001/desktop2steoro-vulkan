from __future__ import annotations

from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import (
    OneVPLD3D11SurfaceEncoder,
    probe_onevpl_d3d11,
)


def test_onevpl_probe_is_safe_without_optional_dll() -> None:
    result = probe_onevpl_d3d11()
    assert result["backend"] == "onevpl_d3d11_surface"
    assert isinstance(result["available"], bool)


def test_onevpl_encoder_requires_optional_bridge() -> None:
    result = probe_onevpl_d3d11()
    if result["available"]:
        return
    try:
        OneVPLD3D11SurfaceEncoder(
            width=64,
            height=64,
            fps=30,
            bitrate=1_000_000,
            d3d11_device=0,
        )
    except RuntimeError as exc:
        assert "onevpl" in str(exc).lower()
    else:
        raise AssertionError("oneVPL encoder unexpectedly initialized without bridge")
