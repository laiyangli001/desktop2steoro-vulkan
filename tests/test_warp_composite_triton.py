from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warp_composite_rgba_u8_matches_float_warp_then_pack():
    pytest.importorskip("triton")
    from stereo_runtime.output_triton import make_chw_rgb_to_hwc_rgba_u8
    from stereo_runtime.warp_composite_triton import warp_composite2, warp_composite2_rgba_u8

    torch.manual_seed(1234)
    rgb = torch.rand((1, 3, 31, 47), device="cuda", dtype=torch.float32)
    depth = torch.rand((1, 1, 31, 47), device="cuda", dtype=torch.float32)
    base_shift = (torch.rand((1, 1, 31, 47), device="cuda", dtype=torch.float32) - 0.5) * 14.0

    float_left, float_right = warp_composite2(rgb, depth, base_shift)
    expected_left = make_chw_rgb_to_hwc_rgba_u8(float_left)
    expected_right = make_chw_rgb_to_hwc_rgba_u8(float_right)
    actual_left, actual_right = warp_composite2_rgba_u8(rgb, depth, base_shift)
    torch.cuda.synchronize()

    assert torch.equal(actual_left, expected_left)
    assert torch.equal(actual_right, expected_right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_direct_half_sbs_matches_openxr_lanczos2_pack():
    pytest.importorskip("triton")
    from stereo_runtime.output import make_sbs
    from stereo_runtime.warp_composite_triton import (
        warp_composite2,
        warp_composite2_half_sbs,
    )

    torch.manual_seed(4321)
    rgb = torch.rand((1, 3, 24, 40), device="cuda", dtype=torch.float32)
    depth = torch.rand((1, 1, 24, 40), device="cuda", dtype=torch.float32)
    base_shift = (
        torch.rand((1, 1, 24, 40), device="cuda", dtype=torch.float32) - 0.5
    ) * 12.0

    left, right = warp_composite2(rgb, depth, base_shift)
    expected = make_sbs(left, right, "half_sbs", fused=True)
    actual = warp_composite2_half_sbs(rgb, depth, base_shift)
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
