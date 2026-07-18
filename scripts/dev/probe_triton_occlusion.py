from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import triton
import triton.language as tl

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_runtime.io import load_rgb
from stereo_runtime.occlusion import make_occlusion_mask
from stereo_runtime.output import match_depth


@triton.jit
def _occlusion_kernel(
    depth,
    shift_abs,
    out,
    amax_shift: tl.constexpr,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    y = offsets // width
    x = offsets - y * width

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


def triton_occlusion(depth: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
    depth = depth.contiguous()
    shift_abs = shift_px.abs().contiguous()
    out = torch.empty_like(depth)
    _, _, height, width = depth.shape
    total = height * width
    amax_shift = float(shift_abs.amax().clamp_min(1e-6).item())
    block = 256
    grid = (triton.cdiv(total, block),)
    _occlusion_kernel[grid](depth, shift_abs, out, amax_shift, total, width, height, block)
    return out


def time_ms(fn, warmup: int = 5, iters: int = 20) -> tuple[float, float, float]:
    for _ in range(warmup):
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return sum(values) / len(values), min(values), max(values)


def main() -> None:
    device = torch.device("cuda")
    rgb = load_rgb("4K.jpg", device=device)
    provider = create_depth_provider(DepthProviderConfig(backend="tensorrt_native", device=device))
    provider.load()
    depth = match_depth(provider.predict(rgb), rgb.shape[-2], rgb.shape[-1])
    shift = compute_shift_px(depth, rgb.shape[-1], ShiftParams())

    expected = make_occlusion_mask(depth, shift)
    actual = triton_occlusion(depth, shift)
    torch.cuda.synchronize()
    diff = (expected - actual).abs()
    print("maxdiff", diff.max().item())
    print("mean diff", diff.mean().item())
    print("changed", int((diff > 0).sum().item()), "/", diff.numel())

    eager = time_ms(lambda: make_occlusion_mask(depth, shift))
    fused = time_ms(lambda: triton_occlusion(depth, shift))
    print(f"eager mean/min/max: {eager[0]:.3f} {eager[1]:.3f} {eager[2]:.3f}")
    print(f"triton mean/min/max: {fused[0]:.3f} {fused[1]:.3f} {fused[2]:.3f}")


if __name__ == "__main__":
    main()
