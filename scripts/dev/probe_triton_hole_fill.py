from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import triton
import triton.language as tl

from stereo_runtime.hole_fill import edge_aware_fill


@triton.jit
def hole_fill_kernel(image, mask, out, total: tl.constexpr, width: tl.constexpr, height: tl.constexpr, pixels: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    batch_channel = offsets // pixels
    batch = batch_channel // 3
    mask_base = batch * pixels

    acc = tl.zeros((block,), tl.float32)
    for dy in tl.static_range(-3, 4):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-3, 4):
            xx = x + dx
            valid = active & valid_y & (xx >= 0) & (xx < width)
            sample_offset = batch_channel * pixels + yy * width + xx
            acc += tl.load(image + sample_offset, mask=valid, other=0.0)

    blurred = acc / 49.0
    value = tl.load(image + offsets, mask=active, other=0.0)
    blend = tl.load(mask + mask_base + pixel, mask=active, other=0.0)
    result = value + (blurred - value) * blend
    tl.store(out + offsets, result, mask=active)


def triton_hole_fill(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    image = image.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(image)
    _, _, height, width = image.shape
    total = image.numel()
    pixels = height * width
    block = 256
    grid = (triton.cdiv(total, block),)
    hole_fill_kernel[grid](image, mask, out, total, width, height, pixels, block)
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
    image = torch.rand(2, 3, 2160, 3840, device=device)
    mask = (torch.rand(2, 1, 2160, 3840, device=device) > 0.8).float()

    expected = edge_aware_fill(image, mask, radius=3, strength=1.0)
    actual = triton_hole_fill(image, mask)
    torch.cuda.synchronize()
    print("maxdiff", (expected - actual).abs().max().item())

    eager = time_ms(lambda: edge_aware_fill(image, mask, radius=3, strength=1.0))
    fused = time_ms(lambda: triton_hole_fill(image, mask))
    print(f"eager mean/min/max: {eager[0]:.3f} {eager[1]:.3f} {eager[2]:.3f}")
    print(f"triton mean/min/max: {fused[0]:.3f} {fused[1]:.3f} {fused[2]:.3f}")


if __name__ == "__main__":
    main()
