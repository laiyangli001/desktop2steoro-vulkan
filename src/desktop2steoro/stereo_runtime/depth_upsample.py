from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from .output import ensure_b1hw, ensure_bchw

DepthUpsampleMode = Literal["bilinear", "guided"]


def upsample_depth(
    depth: torch.Tensor,
    height: int,
    width: int,
    *,
    rgb: torch.Tensor | None = None,
    mode: DepthUpsampleMode = "bilinear",
    edge_strength: float = 0.35,
) -> torch.Tensor:
    """Upsample normalized depth to the RGB frame size.

    `bilinear` preserves current behavior. `guided` keeps the same output
    resolution but softly pulls high-gradient RGB edges from a nearest path to
    reduce bleeding around silhouettes without changing inference resolution.
    """

    depth = ensure_b1hw(depth).float()
    if depth.shape[-2:] == (height, width):
        return depth

    bilinear = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    if mode == "bilinear":
        return bilinear
    if mode != "guided":
        raise ValueError(f"unknown depth upsample mode: {mode!r}")
    if rgb is None:
        return bilinear

    rgb = ensure_bchw(rgb, name="rgb").to(device=bilinear.device).float()
    if rgb.shape[-2:] != (height, width):
        rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)
    rgb = rgb.clamp(0, 1)

    nearest = F.interpolate(depth, size=(height, width), mode="nearest")
    luma = rgb.mean(dim=1, keepdim=True)
    dx = F.pad((luma[..., :, 1:] - luma[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((luma[..., 1:, :] - luma[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = (dx + dy).clamp(0, 1)
    edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)
    weight = (edge * float(edge_strength)).clamp(0, 1)
    return torch.lerp(bilinear, nearest, weight).clamp(0, 1)
