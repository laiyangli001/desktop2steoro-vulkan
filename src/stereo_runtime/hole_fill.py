from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .output import ensure_bchw, ensure_b1hw


def box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius, count_include_pad=True)


def edge_aware_fill(
    image: torch.Tensor,
    mask: torch.Tensor,
    radius: int = 3,
    strength: float = 1.0,
    fused: bool = True,
    mask_feather_radius: int = 0,
) -> torch.Tensor:
    image = ensure_bchw(image, name="image").float()
    mask = ensure_b1hw(mask).to(device=image.device, dtype=image.dtype).clamp(0, 1)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    backend = edge_aware_fill_backend(image, mask, radius=radius, strength=strength, fused=fused, mask_feather_radius=mask_feather_radius)
    if backend == "triton_radius1_feather1":
        from .hole_fill_triton import edge_aware_fill_radius1_strength060_feather1

        return edge_aware_fill_radius1_strength060_feather1(image.contiguous(), mask.contiguous())
    if mask_feather_radius > 0:
        mask = box_blur(mask, radius=int(mask_feather_radius)).clamp(0, 1)
    if backend == "triton_radius1":
        from .hole_fill_triton import edge_aware_fill_radius1_strength060

        return edge_aware_fill_radius1_strength060(image.contiguous(), mask.contiguous())
    if backend == "triton_radius3":
        from .hole_fill_triton import edge_aware_fill_radius3

        return edge_aware_fill_radius3(image.contiguous(), mask.contiguous())
    blurred = box_blur(image, radius=radius)
    blend = (mask * strength).clamp(0, 1)
    return torch.lerp(image, blurred, blend)


def _match_aux_to_image(aux: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
    if aux is None:
        return None
    aux = ensure_b1hw(aux).to(device=image.device, dtype=image.dtype)
    if aux.shape[-2:] != image.shape[-2:]:
        aux = F.interpolate(aux, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if aux.shape[0] == 1 and image.shape[0] > 1:
        aux = aux.expand(image.shape[0], -1, -1, -1)
    return aux


def _sample_horizontal(image: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0:
        return image
    out = torch.empty_like(image)
    if offset > 0:
        out[..., :, :-offset] = image[..., :, offset:]
        out[..., :, -offset:] = image[..., :, -1:]
    else:
        offset = abs(offset)
        out[..., :, offset:] = image[..., :, :-offset]
        out[..., :, :offset] = image[..., :, :1]
    return out


def _directional_average(image: torch.Tensor, radius: int, direction: torch.Tensor) -> torch.Tensor:
    radius = max(1, int(radius))
    left = torch.zeros_like(image)
    right = torch.zeros_like(image)
    for step in range(1, radius + 1):
        left = left + _sample_horizontal(image, -step)
        right = right + _sample_horizontal(image, step)
    left = left / float(radius)
    right = right / float(radius)
    return torch.where(direction < 0, left, right)


def _ui_text_protection_mask(image: torch.Tensor, depth: torch.Tensor | None) -> torch.Tensor:
    luma = image.mean(dim=1, keepdim=True)
    rgb_edge = torch.zeros_like(luma)
    rgb_edge[..., :, 1:] = torch.maximum(rgb_edge[..., :, 1:], (luma[..., :, 1:] - luma[..., :, :-1]).abs())
    rgb_edge[..., 1:, :] = torch.maximum(rgb_edge[..., 1:, :], (luma[..., 1:, :] - luma[..., :-1, :]).abs())
    protect = ((rgb_edge - 0.20) / 0.30).clamp(0, 1)
    if depth is not None:
        depth_edge = torch.zeros_like(protect)
        depth_edge[..., :, 1:] = torch.maximum(depth_edge[..., :, 1:], (depth[..., :, 1:] - depth[..., :, :-1]).abs())
        depth_edge[..., 1:, :] = torch.maximum(depth_edge[..., 1:, :], (depth[..., 1:, :] - depth[..., :-1, :]).abs())
        protect = torch.maximum(protect, ((depth_edge - 0.04) / 0.12).clamp(0, 1) * 0.5)
    return protect


def directional_edge_aware_fill(
    image: torch.Tensor,
    mask: torch.Tensor,
    depth: torch.Tensor | None,
    shift_px: torch.Tensor | None,
    radius: int = 3,
    strength: float = 1.0,
    mask_feather_radius: int = 0,
    depth_edge_threshold: float = 0.03,
    shift_edge_threshold_px: float = 0.05,
) -> torch.Tensor:
    image = ensure_bchw(image, name="image").float()
    mask = ensure_b1hw(mask).to(device=image.device, dtype=image.dtype).clamp(0, 1)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    if mask_feather_radius > 0:
        mask = box_blur(mask, radius=int(mask_feather_radius)).clamp(0, 1)

    depth = _match_aux_to_image(depth, image)
    shift_px = _match_aux_to_image(shift_px, image)
    if depth is None:
        blurred = box_blur(image, radius=radius)
        blend = (mask * strength).clamp(0, 1)
        return torch.lerp(image, blurred, blend)

    left_depth = _sample_horizontal(depth, -1)
    right_depth = _sample_horizontal(depth, 1)
    depth_delta = right_depth - left_depth
    reliable = depth_delta.abs() > float(depth_edge_threshold)
    if shift_px is not None:
        left_shift = _sample_horizontal(shift_px, -1)
        right_shift = _sample_horizontal(shift_px, 1)
        reliable = reliable | ((right_shift - left_shift).abs() > float(shift_edge_threshold_px))

    # Lower normalized depth is treated as the background side, reducing foreground color drag into holes.
    direction = torch.where(right_depth < left_depth, torch.ones_like(depth), -torch.ones_like(depth))
    directional = _directional_average(image, radius=radius, direction=direction)
    blurred = box_blur(image, radius=radius)
    content_aware = directional * 0.75 + blurred * 0.25
    fill = torch.where(reliable.expand_as(image), content_aware, blurred)

    protection = _ui_text_protection_mask(image, depth)
    blend = (mask * strength * (1.0 - protection * 0.70)).clamp(0, 1)
    return torch.lerp(image, fill, blend)


def directional_edge_aware_fill_backend() -> str:
    return "torch_directional_content_aware"

def edge_aware_fill_backend(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float, fused: bool = True, mask_feather_radius: int = 0) -> str:
    if not fused or _triton_disabled_by_env():
        return "torch_avg_pool"
    if int(mask_feather_radius) == 1 and _can_use_triton_fill_radius1_feather1(image, mask, radius=radius, strength=strength):
        return "triton_radius1_feather1"
    if _can_use_triton_fill_radius1(image, mask, radius=radius, strength=strength):
        return "triton_radius1"
    if _can_use_triton_fill_radius3(image, mask, radius=radius, strength=strength):
        return "triton_radius3"
    return "torch_avg_pool"


def _can_use_triton_fill_radius1(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    try:
        from .hole_fill_triton import can_use_triton_radius1
    except Exception:
        return False
    return can_use_triton_radius1(image, mask, radius=radius, strength=strength)


def _can_use_triton_fill_radius1_feather1(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    try:
        from .hole_fill_triton import can_use_triton_radius1
    except Exception:
        return False
    return can_use_triton_radius1(image, mask, radius=radius, strength=strength)


def _can_use_triton_fill_radius3(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    try:
        from .hole_fill_triton import can_use_triton_radius3
    except Exception:
        return False
    return can_use_triton_radius3(image, mask, radius=radius, strength=strength)


def _triton_disabled_by_env() -> bool:
    return (
        os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
    )
