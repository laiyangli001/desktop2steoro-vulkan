from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _occlusion_radius2_kernel(
    depth,
    shift_abs,
    out,
    amax_shift_ptr,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    y = offsets // width
    x = offsets - y * width
    amax_shift = tl.maximum(tl.load(amax_shift_ptr), 1.0e-6)

    found = tl.zeros((block,), tl.int1)
    for dy in tl.static_range(-2, 3):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-2, 3):
            xx = x + dx
            valid = active & valid_y & (xx >= 0) & (xx < width)
            center = yy * width + xx
            depth_value = tl.load(depth + center, mask=valid, other=0.0)
            shift_value = tl.load(shift_abs + center, mask=valid, other=0.0) / amax_shift

            valid_right = valid & (xx < width - 1)
            valid_down = valid & (yy < height - 1)
            depth_right = tl.load(depth + center + 1, mask=valid_right, other=depth_value)
            depth_down = tl.load(depth + center + width, mask=valid_down, other=depth_value)
            shift_right = tl.load(shift_abs + center + 1, mask=valid_right, other=shift_value * amax_shift) / amax_shift
            shift_down = tl.load(shift_abs + center + width, mask=valid_down, other=shift_value * amax_shift) / amax_shift

            edge = (tl.abs(depth_right - depth_value) + tl.abs(depth_down - depth_value)) > 0.04
            shift_edge = (tl.abs(shift_right - shift_value) + tl.abs(shift_down - shift_value)) > 0.05
            found = found | (valid & (edge | shift_edge))

    tl.store(out + offsets, found.to(tl.float32), mask=active)


def can_use_triton_occlusion_radius2(
    depth: torch.Tensor,
    shift_px: torch.Tensor,
    *,
    edge_threshold: float,
    dilation: int,
) -> bool:
    return (
        abs(float(edge_threshold) - 0.04) < 1e-6
        and dilation == 2
        and depth.is_cuda
        and shift_px.is_cuda
        and depth.dtype == torch.float32
        and shift_px.dtype == torch.float32
        and depth.ndim == 4
        and shift_px.ndim == 4
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and shift_px.shape == depth.shape
    )


def make_occlusion_mask_radius2(depth: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
    depth = depth.contiguous()
    shift_abs = shift_px.abs().contiguous()
    out = torch.empty_like(depth)
    _, _, height, width = depth.shape
    total = height * width
    amax_shift = shift_abs.amax().clamp_min(1e-6).reshape(1)
    block = 256
    grid = (triton.cdiv(total, block),)
    _occlusion_radius2_kernel[grid](depth, shift_abs, out, amax_shift, total, width, height, block)
    return out

@triton.jit
def _occlusion_radius1_kernel(
    depth,
    shift_abs,
    out,
    amax_shift_ptr,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    edge_threshold: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    y = offsets // width
    x = offsets - y * width
    amax_shift = tl.maximum(tl.load(amax_shift_ptr), 1.0e-6)

    found = tl.zeros((block,), tl.int1)
    for dy in tl.static_range(-1, 2):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-1, 2):
            xx = x + dx
            valid = active & valid_y & (xx >= 0) & (xx < width)
            center = yy * width + xx
            depth_value = tl.load(depth + center, mask=valid, other=0.0)
            shift_value = tl.load(shift_abs + center, mask=valid, other=0.0) / amax_shift

            valid_right = valid & (xx < width - 1)
            valid_down = valid & (yy < height - 1)
            depth_right = tl.load(depth + center + 1, mask=valid_right, other=depth_value)
            depth_down = tl.load(depth + center + width, mask=valid_down, other=depth_value)
            shift_right = tl.load(shift_abs + center + 1, mask=valid_right, other=shift_value * amax_shift) / amax_shift
            shift_down = tl.load(shift_abs + center + width, mask=valid_down, other=shift_value * amax_shift) / amax_shift

            edge = (tl.abs(depth_right - depth_value) + tl.abs(depth_down - depth_value)) > edge_threshold
            shift_edge = (tl.abs(shift_right - shift_value) + tl.abs(shift_down - shift_value)) > 0.05
            found = found | (valid & (edge | shift_edge))

    tl.store(out + offsets, found.to(tl.float32), mask=active)


def can_use_triton_occlusion_radius1(
    depth: torch.Tensor,
    shift_px: torch.Tensor,
    *,
    edge_threshold: float,
    dilation: int,
) -> bool:
    return (
        (abs(float(edge_threshold) - 0.03) < 1e-6 or abs(float(edge_threshold) - 0.04) < 1e-6)
        and dilation == 1
        and depth.is_cuda
        and shift_px.is_cuda
        and depth.dtype == torch.float32
        and shift_px.dtype == torch.float32
        and depth.ndim == 4
        and shift_px.ndim == 4
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and shift_px.shape == depth.shape
    )


def make_occlusion_mask_radius1(depth: torch.Tensor, shift_px: torch.Tensor, *, edge_threshold: float = 0.03) -> torch.Tensor:
    depth = depth.contiguous()
    shift_abs = shift_px.abs().contiguous()
    out = torch.empty_like(depth)
    _, _, height, width = depth.shape
    total = height * width
    amax_shift = shift_abs.amax().clamp_min(1e-6).reshape(1)
    block = 256
    grid = (triton.cdiv(total, block),)
    _occlusion_radius1_kernel[grid](depth, shift_abs, out, amax_shift, total, width, height, float(edge_threshold), block)
    return out
