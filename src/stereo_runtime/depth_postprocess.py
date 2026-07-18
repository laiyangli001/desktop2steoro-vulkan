from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_b1hw


def apply_depth_pop(depth: torch.Tensor, depth_pop: float, mid: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    depth = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    if abs(depth_pop) < eps:
        return depth
    if depth_pop <= -1.0:
        raise ValueError("depth_pop must be greater than -1.0")
    if depth_pop < 0.0:
        strength = min(1.0, max(0.0, -float(depth_pop)))
        centered = depth - float(mid)
        # Negative Depth Pop is a realtime compression control. Avoid torch.pow here:
        # high exponents near -1.0 are extremely slow on 4K CUDA tensors.
        compressed = centered * (1.0 - strength)
        return (float(mid) + compressed).clamp(0.0, 1.0)
    exponent = 1.0 / (1.0 + float(depth_pop))
    centered = depth - float(mid)
    out = float(mid) + torch.sign(centered) * torch.abs(centered).pow(exponent)
    return out.clamp(0.0, 1.0)


def anti_alias_depth(depth: torch.Tensor, strength: float) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if strength <= 0.0:
        return depth
    kernel_size = int(3 * float(strength)) | 1
    if kernel_size < 3:
        return depth
    sigma = max(0.5 * float(strength), 1e-4)
    coords = torch.arange(kernel_size, device=depth.device, dtype=depth.dtype) - kernel_size // 2
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-6)
    out = F.conv2d(depth, kernel.view(1, 1, 1, -1), padding=(0, kernel_size // 2))
    out = F.conv2d(out, kernel.view(1, 1, -1, 1), padding=(kernel_size // 2, 0))
    return out.clamp(0.0, 1.0)


def postprocess_depth(
    depth: torch.Tensor,
    *,
    depth_pop: float = 0.0,
    antialias_strength: float = 0.0,
) -> torch.Tensor:
    out = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    out = apply_depth_pop(out, depth_pop)
    out = anti_alias_depth(out, antialias_strength)
    return out
