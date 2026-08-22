from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_bchw


def estimate_luma_depth(rgb: torch.Tensor, blur_radius: int = 5) -> torch.Tensor:
    """Fast convenience depth fallback for UI testing, not a real depth model."""
    rgb = ensure_bchw(rgb, name="rgb").float().clamp(0, 1)
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    luma = (rgb * weights).sum(dim=1, keepdim=True)

    if blur_radius > 0:
        k = blur_radius * 2 + 1
        kernel = torch.ones(1, 1, k, k, device=rgb.device, dtype=rgb.dtype) / float(k * k)
        luma = F.conv2d(luma, kernel, padding=blur_radius)

    dx = F.pad((luma[..., :, 1:] - luma[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((luma[..., 1:, :] - luma[..., :-1, :]).abs(), (0, 0, 0, 1))
    edges = (dx + dy).clamp(0, 1)
    depth = (0.65 * luma + 0.35 * edges).clamp(0, 1)
    amin = depth.amin(dim=(-2, -1), keepdim=True)
    amax = depth.amax(dim=(-2, -1), keepdim=True)
    return ((depth - amin) / (amax - amin).clamp_min(1e-6)).clamp(0, 1)
