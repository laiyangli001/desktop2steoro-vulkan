from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import triton
import triton.language as tl

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px, warp_horizontal
from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider
from stereo_runtime.io import load_rgb
from stereo_runtime.layers import composite_layers, make_depth_layers
from stereo_runtime.output import match_depth


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
    scale0 = 0.875
    scale1 = 1.0

    left_value = _sample_two_layers(rgb, channel, y, x, shift, -scale0, -scale1, width, pixels, active, w0, w1)
    right_value = _sample_two_layers(rgb, channel, y, x, shift, scale0, scale1, width, pixels, active, w0, w1)
    tl.store(left + offsets, left_value, mask=active)
    tl.store(right + offsets, right_value, mask=active)


@triton.jit
def _sample_two_layers(rgb, channel, y, x, shift, scale0, scale1, width: tl.constexpr, pixels: tl.constexpr, active, w0, w1):
    x0 = x + shift * scale0
    x1 = x + shift * scale1
    v0 = _sample_border_linear(rgb, channel, y, x0, width, pixels, active)
    v1 = _sample_border_linear(rgb, channel, y, x1, width, pixels, active)
    return v0 * w0 + v1 * w1


@triton.jit
def _sample_border_linear(rgb, channel, y, sample_x, width: tl.constexpr, pixels: tl.constexpr, active):
    x_clamped = tl.minimum(tl.maximum(sample_x, 0.0), width - 1.0)
    x0_float = tl.floor(x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = x_clamped - x0_float
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0, mask=active, other=0.0)
    v1 = tl.load(rgb + base + x1, mask=active, other=0.0)
    return v0 + (v1 - v0) * frac


def fused_warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def torch_warp_composite2(rgb: torch.Tensor, depth: torch.Tensor, base_shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = make_depth_layers(depth, layers=2)
    left_layers = []
    right_layers = []
    for idx in range(2):
        layer_shift = base_shift * (0.75 + 0.25 * (idx + 1) / 2)
        left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=-1.0))
        right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=1.0))
    return composite_layers(left_layers, weights), composite_layers(right_layers, weights)


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
    base_shift = compute_shift_px(depth, rgb.shape[-1], ShiftParams())

    expected_left, expected_right = torch_warp_composite2(rgb, depth, base_shift)
    actual_left, actual_right = fused_warp_composite2(rgb, depth, base_shift)
    torch.cuda.synchronize()
    print("left maxdiff", (expected_left - actual_left).abs().max().item())
    print("right maxdiff", (expected_right - actual_right).abs().max().item())
    print("left mean diff", (expected_left - actual_left).abs().mean().item())
    print("right mean diff", (expected_right - actual_right).abs().mean().item())

    eager = time_ms(lambda: torch_warp_composite2(rgb, depth, base_shift))
    fused = time_ms(lambda: fused_warp_composite2(rgb, depth, base_shift))
    print(f"eager mean/min/max: {eager[0]:.3f} {eager[1]:.3f} {eager[2]:.3f}")
    print(f"triton mean/min/max: {fused[0]:.3f} {fused[1]:.3f} {fused[2]:.3f}")


if __name__ == "__main__":
    main()
