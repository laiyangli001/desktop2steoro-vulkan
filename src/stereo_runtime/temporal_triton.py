from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _temporal_masked_kernel(
    left,
    right,
    prev_left,
    prev_right,
    mask,
    out_left,
    out_right,
    alpha: tl.constexpr,
    total: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    blend = tl.load(mask + pixel, mask=active, other=0.0) * alpha

    left_value = tl.load(left + offsets, mask=active, other=0.0)
    right_value = tl.load(right + offsets, mask=active, other=0.0)
    prev_left_value = tl.load(prev_left + offsets, mask=active, other=0.0)
    prev_right_value = tl.load(prev_right + offsets, mask=active, other=0.0)

    tl.store(out_left + offsets, left_value * (1.0 - blend) + prev_left_value * blend, mask=active)
    tl.store(out_right + offsets, right_value * (1.0 - blend) + prev_right_value * blend, mask=active)


def can_use_triton_temporal_masked(
    left: torch.Tensor,
    right: torch.Tensor,
    prev_left: torch.Tensor,
    prev_right: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    return (
        left.is_cuda
        and right.is_cuda
        and prev_left.is_cuda
        and prev_right.is_cuda
        and mask.is_cuda
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and prev_left.dtype == torch.float32
        and prev_right.dtype == torch.float32
        and mask.dtype == torch.float32
        and left.ndim == 4
        and right.shape == left.shape
        and prev_left.shape == left.shape
        and prev_right.shape == right.shape
        and mask.ndim == 4
        and mask.shape[0] == left.shape[0]
        and mask.shape[1] == 1
        and mask.shape[-2:] == left.shape[-2:]
    )


def apply_temporal_masked(
    left: torch.Tensor,
    right: torch.Tensor,
    prev_left: torch.Tensor,
    prev_right: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = left.contiguous()
    right = right.contiguous()
    prev_left = prev_left.contiguous()
    prev_right = prev_right.contiguous()
    mask = mask.contiguous()
    out_left = torch.empty_like(left)
    out_right = torch.empty_like(right)
    pixels = left.shape[-2] * left.shape[-1]
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _temporal_masked_kernel[grid](
        left,
        right,
        prev_left,
        prev_right,
        mask,
        out_left,
        out_right,
        float(alpha),
        total,
        pixels,
        block,
    )
    return out_left, out_right
