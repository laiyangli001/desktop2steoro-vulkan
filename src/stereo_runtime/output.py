from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn.functional as F

OutputFormat = Literal[
    "half_sbs",
    "full_sbs",
    "half_tab",
    "full_tab",
    "mono",
    "depth_map",
    "anaglyph",
    "interleaved",
    "leia",
]
AnaglyphMethod = Literal["red_cyan", "green_magenta", "amber_blue", "gray"]
OUTPUT_FORMAT_CHOICES = (
    "half_sbs",
    "full_sbs",
    "half_tab",
    "full_tab",
    "mono",
    "depth_map",
    "anaglyph",
    "interleaved",
    "leia",
)


def ensure_bchw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"{name} must be CHW or BCHW, got shape {tuple(x.shape)}")


def ensure_b1hw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 2:
        return depth.unsqueeze(0).unsqueeze(0)
    if depth.ndim == 3:
        return depth.unsqueeze(1)
    if depth.ndim == 4 and depth.shape[1] == 1:
        return depth
    raise ValueError(f"depth must be HW, BHW, or B1HW, got shape {tuple(depth.shape)}")


def match_depth(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if depth.shape[-2:] == (height, width):
        return depth
    return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)


def make_sbs(
    left: torch.Tensor,
    right: torch.Tensor,
    output_format: OutputFormat,
    fused: bool = True,
    depth: torch.Tensor | None = None,
    anaglyph_method: AnaglyphMethod = "red_cyan",
) -> torch.Tensor:
    left = ensure_bchw(left, name="left")
    right = ensure_bchw(right, name="right")
    if left.shape != right.shape:
        raise ValueError(f"left and right shapes must match, got {left.shape} and {right.shape}")

    if output_format == "mono":
        return left

    if output_format == "anaglyph":
        if sbs_backend(left, right, output_format, fused=fused, anaglyph_method=anaglyph_method) == "triton_anaglyph":
            from .output_triton import make_anaglyph

            return make_anaglyph(left, right)
        return make_anaglyph_torch(left, right, method=anaglyph_method)

    if output_format == "interleaved":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_interleaved":
            from .output_triton import make_interleaved

            return make_interleaved(left, right)
        out = torch.empty_like(left)
        out[..., 0::2, :] = left[..., 0::2, :]
        out[..., 1::2, :] = right[..., 1::2, :]
        return out

    if output_format == "leia":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_leia":
            from .output_triton import make_leia

            return make_leia(left, right)
        out = torch.empty_like(left)
        out[..., :, 0::2] = left[..., :, 0::2]
        out[..., :, 1::2] = right[..., :, 1::2]
        return out

    if output_format == "depth_map":
        if depth is None:
            raise ValueError("depth_map output requires depth")
        depth = match_depth(depth, left.shape[-2], left.shape[-1])
        if sbs_backend(left, right, output_format, fused=fused, depth=depth) == "triton_depth_map":
            from .output_triton import make_depth_map

            return make_depth_map(depth, left.shape[1])
        return depth.repeat(1, left.shape[1], 1, 1)

    if output_format == "full_sbs":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_full_sbs":
            from .output_triton import make_full_sbs

            return make_full_sbs(left, right)
        return torch.cat([left, right], dim=-1)

    if output_format == "half_sbs":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_half_sbs":
            from .output_triton import make_half_sbs

            return make_half_sbs(left, right)
        h, w = left.shape[-2:]
        left_w = max(1, w // 2)
        right_w = max(1, w - left_w)
        left_half = F.interpolate(left, size=(h, left_w), mode="area")
        right_half = F.interpolate(right, size=(h, right_w), mode="area")
        return torch.cat([left_half, right_half], dim=-1)

    if output_format == "full_tab":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_full_tab":
            from .output_triton import make_full_tab

            return make_full_tab(left, right)
        return torch.cat([left, right], dim=-2)

    if output_format == "half_tab":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_half_tab":
            from .output_triton import make_half_tab

            return make_half_tab(left, right)
        h, w = left.shape[-2:]
        left_h = max(1, h // 2)
        right_h = max(1, h - left_h)
        left_half = F.interpolate(left, size=(left_h, w), mode="area")
        right_half = F.interpolate(right, size=(right_h, w), mode="area")
        return torch.cat([left_half, right_half], dim=-2)

    raise ValueError(f"unknown output_format: {output_format}")


def sbs_backend(
    left: torch.Tensor,
    right: torch.Tensor,
    output_format: OutputFormat,
    fused: bool = True,
    depth: torch.Tensor | None = None,
    anaglyph_method: AnaglyphMethod = "red_cyan",
) -> str:
    if output_format in {"full_sbs", "full_tab"} and (not fused or _triton_disabled_by_env()):
        return "torch_cat"
    if output_format in {"half_sbs", "half_tab"} and (not fused or _triton_disabled_by_env()):
        return "torch_interpolate"
    if output_format == "depth_map" and (not fused or _triton_disabled_by_env()):
        return "torch_depth_map"
    if output_format == "mono":
        return "torch_mono_left"
    if output_format in {"anaglyph", "interleaved", "leia"} and (not fused or _triton_disabled_by_env()):
        return f"torch_{output_format}"
    if output_format not in {"half_sbs", "full_sbs", "half_tab", "full_tab", "depth_map", "anaglyph", "interleaved", "leia"}:
        return "torch_output"
    try:
        from .output_triton import (
            can_use_triton_anaglyph,
            can_use_triton_depth_map,
            can_use_triton_full_sbs,
            can_use_triton_full_tab,
            can_use_triton_half_sbs,
            can_use_triton_half_tab,
            can_use_triton_interleaved,
            can_use_triton_leia,
        )
    except Exception:
        if output_format in {"anaglyph", "interleaved", "leia"}:
            return f"torch_{output_format}"
        if output_format in {"full_sbs", "full_tab"}:
            return "torch_cat"
        if output_format == "depth_map":
            return "torch_depth_map"
        return "torch_interpolate"
    if output_format == "full_sbs":
        return "triton_full_sbs" if can_use_triton_full_sbs(left, right) else "torch_cat"
    if output_format == "half_sbs":
        return "triton_half_sbs" if can_use_triton_half_sbs(left, right) else "torch_interpolate"
    if output_format == "full_tab":
        return "triton_full_tab" if can_use_triton_full_tab(left, right) else "torch_cat_vertical"
    if output_format == "half_tab":
        return "triton_half_tab" if can_use_triton_half_tab(left, right) else "torch_interpolate_vertical"
    if output_format == "depth_map" and depth is not None:
        return "triton_depth_map" if can_use_triton_depth_map(depth, left.shape[1]) else "torch_depth_map"
    if output_format == "anaglyph":
        return "triton_anaglyph" if anaglyph_method == "red_cyan" and can_use_triton_anaglyph(left, right) else "torch_anaglyph"
    if output_format == "interleaved":
        return "triton_interleaved" if can_use_triton_interleaved(left, right) else "torch_interleaved"
    if output_format == "leia":
        return "triton_leia" if can_use_triton_leia(left, right) else "torch_leia"
    return "torch_depth_map"


def _triton_disabled_by_env() -> bool:
    return (
        os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
    )


def make_anaglyph_torch(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    method: AnaglyphMethod = "red_cyan",
) -> torch.Tensor:
    out = torch.empty_like(left)
    if method == "red_cyan":
        out[:, 0:1] = left[:, 0:1]
        out[:, 1:] = right[:, 1:]
        return out
    if method == "green_magenta":
        out[:, 0:1] = right[:, 0:1]
        out[:, 1:2] = left[:, 1:2]
        out[:, 2:3] = right[:, 2:3]
        return out
    if method == "amber_blue":
        out[:, 0:2] = left[:, 0:2]
        out[:, 2:3] = right[:, 2:3]
        return out
    if method == "gray":
        left_gray = left.mean(dim=1, keepdim=True)
        right_gray = right.mean(dim=1, keepdim=True)
        out[:, 0:1] = left_gray
        out[:, 1:] = right_gray
        return out
    raise ValueError(f"unknown anaglyph_method: {method}")


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clamp(0, 1)
    return (x * 255.0).round().to(torch.uint8)
