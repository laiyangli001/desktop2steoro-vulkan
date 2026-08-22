from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_b1hw

_LAYER_CENTERS_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _layer_centers(layers: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (layers, str(device), dtype)
    centers = _LAYER_CENTERS_CACHE.get(key)
    if centers is None:
        centers = torch.linspace(0.0, 1.0, layers, device=device, dtype=dtype).view(1, layers, 1, 1)
        _LAYER_CENTERS_CACHE[key] = centers
    return centers


def make_depth_layers(depth: torch.Tensor, layers: int = 2, softness: float = 0.08) -> torch.Tensor:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    depth = ensure_b1hw(depth).clamp(0, 1)
    if layers == 1:
        return torch.ones_like(depth)

    centers = _layer_centers(layers, depth.device, depth.dtype)
    depth_blhw = depth.expand(-1, layers, -1, -1)
    weights = torch.exp(-((depth_blhw - centers) ** 2) / max(softness, 1e-4))
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)


def composite_layers(warped: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    if not warped:
        raise ValueError("warped layer list is empty")
    weights = weights[:, : len(warped)]
    if len(warped) == 1:
        return warped[0] * weights[:, 0:1]
    if len(warped) == 2:
        return warped[0] * weights[:, 0:1] + warped[1] * weights[:, 1:2]
    out = torch.zeros_like(warped[0])
    for idx, layer in enumerate(warped):
        out.addcmul_(layer, weights[:, idx : idx + 1])
    return out


def depth_edges(depth: torch.Tensor, threshold: float = 0.04) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    edges = torch.zeros_like(depth)
    edges[..., :, :-1].add_((depth[..., :, 1:] - depth[..., :, :-1]).abs())
    edges[..., :-1, :].add_((depth[..., 1:, :] - depth[..., :-1, :]).abs())
    return (edges > threshold).float()
