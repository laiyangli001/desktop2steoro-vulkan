from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .output import ensure_bchw, match_depth
from .parallax import parallax_debug_info, resolve_parallax_budget


@dataclass(frozen=True)
class ShiftParams:
    depth_strength: float = 2.0
    convergence: float | torch.Tensor = 0.0
    max_disparity_px: float | None = None
    parallax_preset: str = "standard"
    foreground_shift_scale: float = 1.0
    midground_shift_scale: float = 1.0
    background_shift_scale: float = 1.0


_GRID_CACHE: dict[tuple[int, int, int, str, torch.dtype], torch.Tensor] = {}
_GRID_COMPONENT_CACHE: dict[tuple[int, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def make_base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (batch, height, width, str(device), dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    grid = grid.unsqueeze(0).expand(batch, height, width, 2)
    _GRID_CACHE[key] = grid
    return grid


def make_base_grid_components(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, str(device), dtype)
    cached = _GRID_COMPONENT_CACHE.get(key)
    if cached is not None:
        return cached
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    _GRID_COMPONENT_CACHE[key] = (xx, yy)
    return xx, yy


def _layered_shift_response(depth: torch.Tensor, response: torch.Tensor, params: ShiftParams) -> torch.Tensor:
    fg = max(0.0, float(params.foreground_shift_scale))
    mg = max(0.0, float(params.midground_shift_scale))
    bg = max(0.0, float(params.background_shift_scale))
    if abs(fg - 1.0) < 1e-6 and abs(mg - 1.0) < 1e-6 and abs(bg - 1.0) < 1e-6:
        return response
    normalized = depth.clamp(0.0, 1.0)
    background_weight = ((0.5 - normalized) * 2.0).clamp(0.0, 1.0)
    foreground_weight = ((normalized - 0.5) * 2.0).clamp(0.0, 1.0)
    midground_weight = (1.0 - background_weight - foreground_weight).clamp(0.0, 1.0)
    scale = background_weight * bg + midground_weight * mg + foreground_weight * fg
    return response * scale


def compute_shift_px(depth: torch.Tensor, width: int, params: ShiftParams) -> torch.Tensor:
    height = int(depth.shape[-2]) if getattr(depth, "ndim", 0) >= 2 else 1
    budget = resolve_parallax_budget(
        render_width=width,
        render_height=height,
        preset=params.parallax_preset,
        convergence=params.convergence,
        max_disparity_px=params.max_disparity_px,
    )
    depth_strength = max(0.0, float(params.depth_strength))
    response = _layered_shift_response(depth, budget.depth_response(depth), params)
    return -response * depth_strength * budget.max_disparity_px * 0.5


def shift_debug_info(depth: torch.Tensor, width: int, params: ShiftParams) -> dict[str, float | int | str]:
    height = int(depth.shape[-2]) if getattr(depth, "ndim", 0) >= 2 else 1
    budget = resolve_parallax_budget(
        render_width=width,
        render_height=height,
        preset=params.parallax_preset,
        convergence=params.convergence,
        max_disparity_px=params.max_disparity_px,
    )
    debug = parallax_debug_info(budget)
    debug.update(
        {
            "foreground_shift_scale": float(params.foreground_shift_scale),
            "midground_shift_scale": float(params.midground_shift_scale),
            "background_shift_scale": float(params.background_shift_scale),
        }
    )
    return debug


def warp_horizontal(rgb: torch.Tensor, shift_px: torch.Tensor, eye_sign: float) -> torch.Tensor:
    rgb = ensure_bchw(rgb, name="rgb").float()
    b, _, h, w = rgb.shape
    shift_px = match_depth(shift_px, h, w)
    xx, yy = make_base_grid_components(h, w, rgb.device, rgb.dtype)
    shift_norm = (2.0 * shift_px.squeeze(1) / max(w - 1, 1)) * eye_sign
    grid_x = xx.unsqueeze(0) + shift_norm
    grid_y = yy.expand(b, h, w)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def synthesize_baseline(rgb: torch.Tensor, depth: torch.Tensor, params: ShiftParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = match_depth(depth, rgb.shape[-2], rgb.shape[-1])
    shift_px = compute_shift_px(depth, rgb.shape[-1], params)
    left = warp_horizontal(rgb, shift_px, eye_sign=1.0)
    right = warp_horizontal(rgb, shift_px, eye_sign=-1.0)
    return left, right, shift_px
