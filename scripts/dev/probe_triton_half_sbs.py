from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import triton
import triton.language as tl

from stereo_runtime.output import make_sbs


@triton.jit
def _half_sbs_kernel(left, right, out, total: tl.constexpr, width: tl.constexpr, half_width: tl.constexpr, height: tl.constexpr, pixels: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = x < half_width
    src_x_out = tl.where(use_left, x, x - half_width)
    # F.interpolate(..., align_corners=False): in_x = (out_x + 0.5) * in_w / out_w - 0.5
    src_x = (src_x_out.to(tl.float32) + 0.5) * (width / half_width) - 0.5
    src_x_clamped = tl.minimum(tl.maximum(src_x, 0.0), width - 1.0)
    x0_float = tl.floor(src_x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = src_x_clamped - x0_float
    base = channel * pixels + y * width
    left_v0 = tl.load(left + base + x0, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + base + x1, mask=active & use_left, other=0.0)
    right_v0 = tl.load(right + base + x0, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + base + x1, mask=active & ~use_left, other=0.0)
    value = tl.where(use_left, left_v0 + (left_v1 - left_v0) * frac, right_v0 + (right_v1 - right_v0) * frac)
    tl.store(out + offsets, value, mask=active)


def triton_half_sbs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    half_width = width // 2
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_sbs_kernel[grid](left, right, out, total, width, half_width, height, pixels, block)
    return out


def time_ms(fn, warmup: int = 5, iters: int = 30) -> tuple[float, float, float]:
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
    left = torch.rand(1, 3, 2160, 3840, device="cuda")
    right = torch.rand(1, 3, 2160, 3840, device="cuda")
    expected = make_sbs(left, right, "half_sbs", fused=False)
    actual = triton_half_sbs(left, right)
    torch.cuda.synchronize()
    diff = (expected - actual).abs()
    print("maxdiff", diff.max().item())
    print("mean diff", diff.mean().item())
    print("changed", int((diff > 1e-6).sum().item()), "/", diff.numel())
    eager = time_ms(lambda: make_sbs(left, right, "half_sbs", fused=False))
    fused = time_ms(lambda: triton_half_sbs(left, right))
    print(f"eager mean/min/max: {eager[0]:.3f} {eager[1]:.3f} {eager[2]:.3f}")
    print(f"triton mean/min/max: {fused[0]:.3f} {fused[1]:.3f} {fused[2]:.3f}")


if __name__ == "__main__":
    main()
