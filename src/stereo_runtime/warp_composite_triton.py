from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _warp_composite2_kernel(
    rgb,
    depth,
    base_shift,
    left,
    right,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    softness: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    depth_value = tl.load(depth + pixel, mask=active, other=0.0)
    w0_raw = tl.exp(-((depth_value - 0.0) * (depth_value - 0.0)) / softness)
    w1_raw = tl.exp(-((depth_value - 1.0) * (depth_value - 1.0)) / softness)
    wsum = w0_raw + w1_raw
    w0 = w0_raw / wsum
    w1 = w1_raw / wsum

    shift = tl.load(base_shift + pixel, mask=active, other=0.0)
    left_value = _sample_two_layers(rgb, channel, y, x, shift, 0.875, 1.0, width, pixels, active, w0, w1)
    right_value = _sample_two_layers(rgb, channel, y, x, shift, -0.875, -1.0, width, pixels, active, w0, w1)
    tl.store(left + offsets, left_value, mask=active)
    tl.store(right + offsets, right_value, mask=active)


@triton.jit
def _sample_two_layers(rgb, channel, y, x, shift, scale0, scale1, width: tl.constexpr, pixels: tl.constexpr, active, w0, w1):
    x0 = x + shift * scale0
    x1 = x + shift * scale1
    v0 = _sample_reflection_linear(rgb, channel, y, x0, width, pixels, active)
    v1 = _sample_reflection_linear(rgb, channel, y, x1, width, pixels, active)
    return v0 * w0 + v1 * w1


@triton.jit
def _sample_reflection_linear(rgb, channel, y, sample_x, width: tl.constexpr, pixels: tl.constexpr, active):
    right_edge = width - 1.0
    x_reflected = tl.where(sample_x < 0.0, -sample_x, sample_x)
    x_reflected = tl.where(x_reflected > right_edge, 2.0 * right_edge - x_reflected, x_reflected)
    x_reflected = tl.minimum(tl.maximum(x_reflected, 0.0), right_edge)
    x0_float = tl.floor(x_reflected)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = x_reflected - x0_float
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0, mask=active, other=0.0)
    v1 = tl.load(rgb + base + x1, mask=active, other=0.0)
    return v0 + (v1 - v0) * frac


def can_use_triton_warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor, *, layers: int, symmetric: bool) -> bool:
    return (
        layers == 2
        and symmetric
        and rgb.is_cuda
        and depth.is_cuda
        and base_shift.is_cuda
        and rgb.dtype == torch.float32
        and depth.dtype == torch.float32
        and base_shift.dtype == torch.float32
        and rgb.ndim == 4
        and depth.ndim == 4
        and base_shift.ndim == 4
        and rgb.shape[0] == 1
        and rgb.shape[1] == 3
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and base_shift.shape == depth.shape
        and rgb.shape[-2:] == depth.shape[-2:]
    )


def warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    base_shift = base_shift.contiguous()
    left = torch.empty_like(rgb)
    right = torch.empty_like(rgb)
    _, _, height, width = rgb.shape
    pixels = height * width
    total = rgb.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _warp_composite2_kernel[grid](rgb, depth, base_shift, left, right, total, width, height, pixels, 0.08, block)
    return left, right
