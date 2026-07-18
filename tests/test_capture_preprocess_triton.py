import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capture.preprocess_triton import bgr_to_rgb_resize_norm, can_use_triton_preprocess


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _torch_reference(frame_raw, out_height, out_width):
    frame_rgb = frame_raw[..., [2, 1, 0]].permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
    if frame_rgb.shape[-2:] == (out_height, out_width):
        return frame_rgb
    return F.interpolate(
        frame_rgb.unsqueeze(0),
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).squeeze(0)


@pytest.mark.parametrize("channels", [3, 4])
def test_triton_preprocess_matches_torch_bilinear(channels):
    frame = torch.arange(7 * 9 * channels, device="cuda", dtype=torch.uint8).reshape(7, 9, channels)

    actual = bgr_to_rgb_resize_norm(frame, 5, 6)
    expected = _torch_reference(frame, 5, 6)

    assert can_use_triton_preprocess(frame)
    assert actual.shape == (3, 5, 6)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)