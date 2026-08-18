from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised by the Torch fallback
    triton = None
    tl = None

from .triton_runtime import triton_runtime_available


if triton is not None:

    @triton.jit
    def _layered_shift_kernel(
        depth_ptr,
        output_ptr,
        total: tl.constexpr,
        convergence: tl.constexpr,
        output_scale: tl.constexpr,
        foreground_scale: tl.constexpr,
        midground_scale: tl.constexpr,
        background_scale: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        depth = tl.load(depth_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        depth = tl.maximum(0.0, tl.minimum(1.0, depth))
        background = background_scale + (2.0 * depth) * (midground_scale - background_scale)
        foreground = midground_scale + (2.0 * depth - 1.0) * (foreground_scale - midground_scale)
        layer_scale = tl.where(depth < 0.5, background, foreground)
        shift = (depth - convergence) * layer_scale * output_scale
        tl.store(output_ptr + offsets, shift, mask=mask)


def can_use_triton_layered_shift(depth: torch.Tensor, convergence: object) -> bool:
    return bool(
        triton is not None
        and isinstance(depth, torch.Tensor)
        and depth.is_cuda
        and depth.dtype == torch.float32
        and depth.is_contiguous()
        and not isinstance(convergence, torch.Tensor)
        and triton_runtime_available(depth.device)
    )


def compute_layered_shift(
    depth: torch.Tensor,
    *,
    convergence: float,
    output_scale: float,
    foreground_scale: float,
    midground_scale: float,
    background_scale: float,
) -> torch.Tensor:
    if not can_use_triton_layered_shift(depth, convergence):
        raise ValueError("Triton layered shift requires contiguous CUDA float32 depth and scalar convergence")
    output = torch.empty_like(depth)
    total = depth.numel()
    block = 256
    _layered_shift_kernel[(triton.cdiv(total, block),)](
        depth,
        output,
        total=total,
        convergence=float(convergence),
        output_scale=float(output_scale),
        foreground_scale=float(foreground_scale),
        midground_scale=float(midground_scale),
        background_scale=float(background_scale),
        BLOCK=block,
    )
    return output
