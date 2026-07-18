from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hole_fill_radius3_kernel(
    image,
    mask,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
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


def can_use_triton_radius3(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    return (
        radius == 3
        and strength == 1.0
        and image.is_cuda
        and mask.is_cuda
        and image.dtype == torch.float32
        and mask.dtype == torch.float32
        and image.ndim == 4
        and mask.ndim == 4
        and image.shape[1] == 3
        and mask.shape[1] == 1
        and image.shape[0] == mask.shape[0]
        and image.shape[-2:] == mask.shape[-2:]
    )


def edge_aware_fill_radius3(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(image)
    _, _, height, width = image.shape
    total = image.numel()
    pixels = height * width
    block = 256
    grid = (triton.cdiv(total, block),)
    _hole_fill_radius3_kernel[grid](image, mask, out, total, width, height, pixels, block)
    return out

@triton.jit
def _hole_fill_radius1_strength060_kernel(
    image,
    mask,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    batch_channel = offsets // pixels
    batch = batch_channel // 3
    mask_base = batch * pixels

    acc = tl.zeros((block,), tl.float32)
    count = tl.zeros((block,), tl.float32)
    for dy in tl.static_range(-1, 2):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-1, 2):
            xx = x + dx
            valid = active & valid_y & (xx >= 0) & (xx < width)
            sample_offset = batch_channel * pixels + yy * width + xx
            acc += tl.load(image + sample_offset, mask=valid, other=0.0)
            count += valid.to(tl.float32)

    blurred = acc / tl.maximum(count, 1.0)
    value = tl.load(image + offsets, mask=active, other=0.0)
    blend = tl.load(mask + mask_base + pixel, mask=active, other=0.0) * 0.60
    result = value + (blurred - value) * blend
    tl.store(out + offsets, result, mask=active)


def can_use_triton_radius1(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    return (
        radius == 1
        and abs(float(strength) - 0.60) < 1e-6
        and image.is_cuda
        and mask.is_cuda
        and image.dtype == torch.float32
        and mask.dtype == torch.float32
        and image.ndim == 4
        and mask.ndim == 4
        and image.shape[1] == 3
        and mask.shape[1] == 1
        and image.shape[0] == mask.shape[0]
        and image.shape[-2:] == mask.shape[-2:]
    )


def edge_aware_fill_radius1_strength060(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(image)
    _, _, height, width = image.shape
    total = image.numel()
    pixels = height * width
    block = 256
    grid = (triton.cdiv(total, block),)
    _hole_fill_radius1_strength060_kernel[grid](image, mask, out, total, width, height, pixels, block)
    return out


@triton.jit
def _hole_fill_radius1_strength060_feather1_kernel(
    image,
    mask,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    batch_channel = offsets // pixels
    batch = batch_channel // 3
    mask_base = batch * pixels

    acc = tl.zeros((block,), tl.float32)
    count = tl.zeros((block,), tl.float32)
    mask_acc = tl.zeros((block,), tl.float32)
    for dy in tl.static_range(-1, 2):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-1, 2):
            xx = x + dx
            valid = active & valid_y & (xx >= 0) & (xx < width)
            sample_pixel = yy * width + xx
            sample_offset = batch_channel * pixels + sample_pixel
            acc += tl.load(image + sample_offset, mask=valid, other=0.0)
            count += valid.to(tl.float32)
            mask_acc += tl.load(mask + mask_base + sample_pixel, mask=valid, other=0.0)

    blurred = acc / tl.maximum(count, 1.0)
    value = tl.load(image + offsets, mask=active, other=0.0)
    feathered_mask = mask_acc / 9.0
    blend = tl.minimum(tl.maximum(feathered_mask, 0.0), 1.0) * 0.60
    result = value + (blurred - value) * blend
    tl.store(out + offsets, result, mask=active)


def edge_aware_fill_radius1_strength060_feather1(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(image)
    _, _, height, width = image.shape
    total = image.numel()
    pixels = height * width
    block = 256
    grid = (triton.cdiv(total, block),)
    _hole_fill_radius1_strength060_feather1_kernel[grid](image, mask, out, total, width, height, pixels, block)
    return out
