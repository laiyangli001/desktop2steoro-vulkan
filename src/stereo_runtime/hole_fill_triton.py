from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_runtime import triton_runtime_available


@triton.jit
def _directional_content_aware_radius3_kernel(
    image,
    mask,
    depth,
    shift_px,
    out,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    strength,
    depth_edge_threshold,
    shift_edge_threshold_px,
    has_shift: tl.constexpr,
    feather_radius: tl.constexpr,
    block: tl.constexpr,
):
    pixel = tl.program_id(0) * block + tl.arange(0, block)
    batch = tl.program_id(1)
    active = pixel < pixels
    y = pixel // width
    x = pixel - y * width

    image_base = batch * 3 * pixels
    aux_base = batch * pixels
    image0 = tl.load(image + image_base + pixel, mask=active, other=0.0)
    image1 = tl.load(image + image_base + pixels + pixel, mask=active, other=0.0)
    image2 = tl.load(image + image_base + 2 * pixels + pixel, mask=active, other=0.0)
    depth_value = tl.load(depth + aux_base + pixel, mask=active, other=0.0)

    x_left = tl.maximum(x - 1, 0)
    x_right = tl.minimum(x + 1, width - 1)
    left_pixel = y * width + x_left
    right_pixel = y * width + x_right
    left_depth = tl.load(depth + aux_base + left_pixel, mask=active, other=0.0)
    right_depth = tl.load(depth + aux_base + right_pixel, mask=active, other=0.0)
    reliable = tl.abs(right_depth - left_depth) > depth_edge_threshold
    if has_shift:
        left_shift = tl.load(shift_px + aux_base + left_pixel, mask=active, other=0.0)
        right_shift = tl.load(shift_px + aux_base + right_pixel, mask=active, other=0.0)
        reliable = reliable | (tl.abs(right_shift - left_shift) > shift_edge_threshold_px)

    use_right = right_depth < left_depth
    directional0 = tl.zeros((block,), tl.float32)
    directional1 = tl.zeros((block,), tl.float32)
    directional2 = tl.zeros((block,), tl.float32)
    for step in tl.static_range(1, 4):
        sample_left_x = tl.maximum(x - step, 0)
        sample_right_x = tl.minimum(x + step, width - 1)
        sample_x = tl.where(use_right, sample_right_x, sample_left_x)
        sample_pixel = y * width + sample_x
        directional0 += tl.load(image + image_base + sample_pixel, mask=active, other=0.0)
        directional1 += tl.load(image + image_base + pixels + sample_pixel, mask=active, other=0.0)
        directional2 += tl.load(image + image_base + 2 * pixels + sample_pixel, mask=active, other=0.0)
    directional0 *= 1.0 / 3.0
    directional1 *= 1.0 / 3.0
    directional2 *= 1.0 / 3.0

    blurred0 = tl.zeros((block,), tl.float32)
    blurred1 = tl.zeros((block,), tl.float32)
    blurred2 = tl.zeros((block,), tl.float32)
    for dy in tl.static_range(-3, 4):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        safe_y = tl.minimum(tl.maximum(yy, 0), height - 1)
        for dx in tl.static_range(-3, 4):
            xx = x + dx
            valid = active & valid_y & (xx >= 0) & (xx < width)
            safe_x = tl.minimum(tl.maximum(xx, 0), width - 1)
            sample_pixel = safe_y * width + safe_x
            blurred0 += tl.load(image + image_base + sample_pixel, mask=valid, other=0.0)
            blurred1 += tl.load(image + image_base + pixels + sample_pixel, mask=valid, other=0.0)
            blurred2 += tl.load(image + image_base + 2 * pixels + sample_pixel, mask=valid, other=0.0)
    blurred0 *= 1.0 / 49.0
    blurred1 *= 1.0 / 49.0
    blurred2 *= 1.0 / 49.0

    content0 = directional0 * 0.75 + blurred0 * 0.25
    content1 = directional1 * 0.75 + blurred1 * 0.25
    content2 = directional2 * 0.75 + blurred2 * 0.25
    fill0 = tl.where(reliable, content0, blurred0)
    fill1 = tl.where(reliable, content1, blurred1)
    fill2 = tl.where(reliable, content2, blurred2)

    luma = (image0 + image1 + image2) * (1.0 / 3.0)
    left0 = tl.load(image + image_base + left_pixel, mask=active, other=0.0)
    left1 = tl.load(image + image_base + pixels + left_pixel, mask=active, other=0.0)
    left2 = tl.load(image + image_base + 2 * pixels + left_pixel, mask=active, other=0.0)
    left_luma = (left0 + left1 + left2) * (1.0 / 3.0)
    rgb_edge_x = tl.where(x > 0, tl.abs(luma - left_luma), 0.0)

    y_up = tl.maximum(y - 1, 0)
    up_pixel = y_up * width + x
    up0 = tl.load(image + image_base + up_pixel, mask=active, other=0.0)
    up1 = tl.load(image + image_base + pixels + up_pixel, mask=active, other=0.0)
    up2 = tl.load(image + image_base + 2 * pixels + up_pixel, mask=active, other=0.0)
    up_luma = (up0 + up1 + up2) * (1.0 / 3.0)
    rgb_edge_y = tl.where(y > 0, tl.abs(luma - up_luma), 0.0)
    rgb_edge = tl.maximum(rgb_edge_x, rgb_edge_y)
    protection = tl.minimum(tl.maximum((rgb_edge - 0.20) / 0.30, 0.0), 1.0)

    left_depth_for_edge = tl.load(depth + aux_base + left_pixel, mask=active, other=0.0)
    up_depth = tl.load(depth + aux_base + up_pixel, mask=active, other=0.0)
    depth_edge_x = tl.where(x > 0, tl.abs(depth_value - left_depth_for_edge), 0.0)
    depth_edge_y = tl.where(y > 0, tl.abs(depth_value - up_depth), 0.0)
    depth_edge = tl.maximum(depth_edge_x, depth_edge_y)
    depth_protection = tl.minimum(tl.maximum((depth_edge - 0.04) / 0.12, 0.0), 1.0) * 0.5
    protection = tl.maximum(protection, depth_protection)

    if feather_radius == 0:
        fill_mask = tl.load(mask + aux_base + pixel, mask=active, other=0.0)
    else:
        mask_acc = tl.zeros((block,), tl.float32)
        for dy in tl.static_range(-feather_radius, feather_radius + 1):
            yy = y + dy
            valid_y = (yy >= 0) & (yy < height)
            safe_y = tl.minimum(tl.maximum(yy, 0), height - 1)
            for dx in tl.static_range(-feather_radius, feather_radius + 1):
                xx = x + dx
                valid = active & valid_y & (xx >= 0) & (xx < width)
                safe_x = tl.minimum(tl.maximum(xx, 0), width - 1)
                sample_pixel = safe_y * width + safe_x
                mask_acc += tl.load(mask + aux_base + sample_pixel, mask=valid, other=0.0)
        feather_width: tl.constexpr = feather_radius * 2 + 1
        fill_mask = mask_acc / float(feather_width * feather_width)

    fill_mask = tl.minimum(tl.maximum(fill_mask, 0.0), 1.0)
    blend = fill_mask * strength * (1.0 - protection * 0.70)
    blend = tl.minimum(tl.maximum(blend, 0.0), 1.0)
    out0 = image0 + (fill0 - image0) * blend
    out1 = image1 + (fill1 - image1) * blend
    out2 = image2 + (fill2 - image2) * blend
    tl.store(out + image_base + pixel, out0, mask=active)
    tl.store(out + image_base + pixels + pixel, out1, mask=active)
    tl.store(out + image_base + 2 * pixels + pixel, out2, mask=active)


def can_use_triton_directional_content_aware(
    image: torch.Tensor,
    mask: torch.Tensor,
    depth: torch.Tensor | None,
    shift_px: torch.Tensor | None,
    *,
    radius: int,
    mask_feather_radius: int,
) -> bool:
    if depth is None:
        return False
    batch, channels, height, width = image.shape if image.ndim == 4 else (0, 0, 0, 0)
    common = (
        radius == 3
        and 0 <= int(mask_feather_radius) <= 3
        and triton_runtime_available(image.device)
        and image.dtype == torch.float32
        and mask.dtype == torch.float32
        and depth.dtype == torch.float32
        and image.ndim == 4
        and mask.ndim == 4
        and depth.ndim == 4
        and channels == 3
        and mask.shape[1] == 1
        and depth.shape[1] == 1
        and mask.shape[0] in {1, batch}
        and depth.shape[0] in {1, batch}
        and image.shape[-2:] == (height, width)
        and mask.shape[-2:] == (height, width)
        and depth.shape[-2:] == (height, width)
    )
    if not common or shift_px is None:
        return common
    return (
        shift_px.dtype == torch.float32
        and shift_px.ndim == 4
        and shift_px.shape[1] == 1
        and shift_px.shape[0] in {1, batch}
        and shift_px.shape[-2:] == (height, width)
    )


def directional_content_aware_fill_radius3(
    image: torch.Tensor,
    mask: torch.Tensor,
    depth: torch.Tensor,
    shift_px: torch.Tensor | None,
    *,
    strength: float,
    mask_feather_radius: int,
    depth_edge_threshold: float,
    shift_edge_threshold_px: float,
) -> torch.Tensor:
    image = image.contiguous()
    batch, _, height, width = image.shape
    mask = mask.expand(batch, -1, -1, -1).contiguous()
    depth = depth.expand(batch, -1, -1, -1).contiguous()
    if shift_px is None:
        shift = depth
        has_shift = False
    else:
        shift = shift_px.expand(batch, -1, -1, -1).contiguous()
        has_shift = True
    out = torch.empty_like(image)
    pixels = height * width
    block = 128
    grid = (triton.cdiv(pixels, block), batch)
    _directional_content_aware_radius3_kernel[grid](
        image,
        mask,
        depth,
        shift,
        out,
        width,
        height,
        pixels,
        float(strength),
        float(depth_edge_threshold),
        float(shift_edge_threshold_px),
        has_shift,
        int(mask_feather_radius),
        block,
    )
    return out


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
        and triton_runtime_available(image.device)
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
        and triton_runtime_available(image.device)
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
