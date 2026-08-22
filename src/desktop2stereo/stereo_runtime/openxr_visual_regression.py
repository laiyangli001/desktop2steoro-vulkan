from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.cpu_warnings import describe_tensor, warn_cpu_operation

from .io import load_depth, load_rgb, save_depth, save_rgb
from .output import ensure_b1hw, ensure_bchw


@dataclass(frozen=True)
class OpenXRViewerShaderParams:
    max_disparity_px: float = 96.0
    convergence: float = 0.0
    screen_roll: float = 0.0
    edge_falloff: bool = True
    shaped_depth: bool = True
    shader_resolution_mode: str = "source"
    swapchain_width: int = 3648
    swapchain_height: int = 3648

    def disparity_uv(self, render_width: int) -> float:
        return max(0.0, float(self.max_disparity_px)) / float(max(1, int(render_width)))


def make_visual_regression_inputs(width: int = 384, height: int = 216) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a high-contrast frame that exposes edge ghosts in DIBR paths."""
    y = torch.linspace(0.0, 1.0, height)
    x = torch.linspace(0.0, 1.0, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rgb = torch.stack(
        [
            0.08 + xx * 0.28,
            0.10 + yy * 0.22,
            torch.full_like(xx, 0.18),
        ],
        dim=0,
    )
    # Foreground bars and a circular object create vertical and curved depth edges.
    fg = ((xx > 0.32) & (xx < 0.48) & (yy > 0.18) & (yy < 0.86)) | (
        ((xx - 0.68) ** 2 + (yy - 0.52) ** 2) < 0.075 ** 2
    )
    rgb[:, fg] = torch.tensor([0.94, 0.92, 0.86]).view(3, 1)
    stripe = ((torch.arange(width).view(1, width) // 8) % 2 == 0).expand(height, width)
    rgb[2, stripe] += 0.18
    rgb = rgb.clamp(0.0, 1.0).unsqueeze(0)
    depth = torch.full((1, 1, height, width), 0.76)
    depth[:, :, fg] = 0.18
    return rgb, depth


def make_depth_proxy_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Create a deterministic edge-rich depth proxy from a real RGB frame."""
    rgb = ensure_bchw(rgb, name="rgb").float().clamp(0.0, 1.0)
    gray = (
        rgb[:, 0:1] * 0.299
        + rgb[:, 1:2] * 0.587
        + rgb[:, 2:3] * 0.114
    )
    blur = F.avg_pool2d(gray, kernel_size=31, stride=1, padding=15)
    quantized = torch.floor((1.0 - blur).mul(7.0)).div(6.0).clamp(0.0, 1.0)

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).view(1, 1, 3, 3)
    edge = (
        F.conv2d(gray, sobel_x, padding=1).abs()
        + F.conv2d(gray, sobel_y, padding=1).abs()
    ).clamp(0.0, 1.0)
    edge = F.max_pool2d(edge, kernel_size=5, stride=1, padding=2)
    depth = (0.18 + quantized * 0.68 - edge * 0.12).clamp(0.0, 1.0)
    return depth.contiguous()


def _shader_resolution(width: int, height: int, params: OpenXRViewerShaderParams) -> tuple[int, int]:
    mode = str(params.shader_resolution_mode or "source").strip().lower()
    if mode == "swapchain":
        return max(1, int(params.swapchain_width)), max(1, int(params.swapchain_height))
    return max(1, int(width)), max(1, int(height))


def _smooth_depth_along_roll(depth: torch.Tensor, params: OpenXRViewerShaderParams, eye_sign: float) -> torch.Tensor:
    b, _, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    c = torch.tensor(float(torch.cos(torch.tensor(params.screen_roll))), device=device, dtype=dtype)
    s = torch.tensor(float(torch.sin(torch.tensor(params.screen_roll))), device=device, dtype=dtype)
    # GLSL uses pixel_size * 1.5 along the signed parallax direction. In OpenXR
    # this can be source texture size or swapchain size depending on uniform setup.
    res_w, res_h = _shader_resolution(w, h, params)
    step_x = 2.0 * 1.5 * c * float(eye_sign) / max(res_w - 1, 1)
    step_y = 2.0 * 1.5 * s * float(eye_sign) / max(res_h - 1, 1)
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, h, w, 2)
    dm = F.grid_sample(depth, base - torch.tensor([step_x, step_y], device=device, dtype=dtype), mode="bilinear", padding_mode="border", align_corners=True)
    dp = F.grid_sample(depth, base + torch.tensor([step_x, step_y], device=device, dtype=dtype), mode="bilinear", padding_mode="border", align_corners=True)
    return depth * 0.5 + dm * 0.25 + dp * 0.25


def render_viewer_shader_eye_cpu(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    eye_sign: float,
    params: OpenXRViewerShaderParams,
) -> torch.Tensor:
    """CPU/Torch approximation of the OpenXR viewer rgb_depth shader path."""
    warn_cpu_operation(
        "OpenXR visual regression",
        "CPU/Torch shader approximation",
        detail=f"rgb={describe_tensor(rgb)} depth={describe_tensor(depth)}",
        key="openxr_visual_regression_cpu_shader",
    )
    rgb = ensure_bchw(rgb, name="rgb").float().clamp(0.0, 1.0)
    depth = ensure_b1hw(depth).float().clamp(0.0, 1.0)
    if depth.shape[-2:] != rgb.shape[-2:]:
        depth = F.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    b, _, h, w = rgb.shape
    device = rgb.device
    dtype = rgb.dtype
    d = _smooth_depth_along_roll(depth.to(device=device, dtype=dtype), params, eye_sign)
    depth_inv = -d
    if params.shaped_depth:
        depth_inv = depth_inv * (1.0 + 0.25 * (1.0 - d))
    shift = depth_inv + float(params.convergence)
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    uv_x = (xx + 1.0) * 0.5
    if params.edge_falloff:
        left = torch.clamp(uv_x / 0.05, 0.0, 1.0)
        right = torch.clamp((1.0 - uv_x) / 0.05, 0.0, 1.0)
        falloff = (left * left * (3.0 - 2.0 * left)) * (right * right * (3.0 - 2.0 * right))
    else:
        falloff = torch.ones_like(xx)
    eye_offset = float(eye_sign) * params.disparity_uv(w) / 2.0
    px_uv = eye_offset * shift.squeeze(1) * falloff.unsqueeze(0)
    c = torch.cos(torch.tensor(float(params.screen_roll), device=device, dtype=dtype))
    s = torch.sin(torch.tensor(float(params.screen_roll), device=device, dtype=dtype))
    grid_x = xx.unsqueeze(0).expand(b, h, w) - px_uv * c * 2.0
    grid_y = yy.unsqueeze(0).expand(b, h, w) - px_uv * s * 2.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True).clamp(0.0, 1.0)


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = ensure_bchw(a, name="a").float()
    b = ensure_bchw(b, name="b").float()
    diff = (a - b).abs()
    warn_cpu_operation(
        "OpenXR visual regression compare_tensors",
        "diff metrics .item() sync",
        detail=describe_tensor(diff),
        key="openxr_visual_regression_compare_cpu_stats",
    )
    return {
        "mae": float(diff.mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "max": float(diff.max().item()),
        "pct_gt_1_255": float((diff > (1.0 / 255.0)).float().mean().item()),
        "pct_gt_5_255": float((diff > (5.0 / 255.0)).float().mean().item()),
    }


def diff_heatmap(a: torch.Tensor, b: torch.Tensor, gain: float = 8.0) -> torch.Tensor:
    diff = (ensure_bchw(a, name="a").float() - ensure_bchw(b, name="b").float()).abs()
    heat = diff.mean(dim=1, keepdim=True).mul(gain).clamp(0.0, 1.0)
    return torch.cat((heat, heat * 0.25, 1.0 - heat), dim=1).clamp(0.0, 1.0)


def run_openxr_visual_regression(
    *,
    output_dir: str | Path,
    rgb_path: str | Path | None = None,
    depth_path: str | Path | None = None,
    params: OpenXRViewerShaderParams = OpenXRViewerShaderParams(),
) -> dict[str, Any]:
    out = Path(output_dir)
    if rgb_path is None:
        rgb, depth = make_visual_regression_inputs()
    else:
        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path) if depth_path is not None else make_depth_proxy_from_rgb(rgb)
    source_params = params
    swapchain_params = OpenXRViewerShaderParams(
        **{**asdict(params), "shader_resolution_mode": "swapchain"}
    )
    outputs: dict[str, torch.Tensor] = {}
    for label, p in (
        ("source", source_params),
        ("swapchain", swapchain_params),
    ):
        outputs[f"{label}_left"] = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=-1.0, params=p)
        outputs[f"{label}_right"] = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=p)
    source_vs_swapchain_metrics = {
        "left": compare_tensors(outputs["source_left"], outputs["swapchain_left"]),
        "right": compare_tensors(outputs["source_right"], outputs["swapchain_right"]),
    }
    metrics = {
        "params": asdict(params),
        "source_vs_swapchain": source_vs_swapchain_metrics,
        "ranking_by_mean_mae": [
            {
                "variant": "source",
                "shader_resolution_mode": "source",
                "mean_mae": 0.0,
            },
            {
                "variant": "swapchain",
                "shader_resolution_mode": "swapchain",
                "mean_mae": (
                    source_vs_swapchain_metrics["left"]["mae"]
                    + source_vs_swapchain_metrics["right"]["mae"]
                ) * 0.5,
            },
        ],
    }
    save_rgb(rgb, out / "source_rgb.png")
    save_rgb(depth.expand(-1, 3, -1, -1), out / "source_depth.png")
    save_depth(depth, out / "prepared_depth.png")
    for name, tensor in outputs.items():
        save_rgb(tensor, out / f"{name}.png")
    save_rgb(diff_heatmap(outputs["source_left"], outputs["swapchain_left"]), out / "diff_source_vs_swapchain_left_heatmap.png")
    save_rgb(diff_heatmap(outputs["source_right"], outputs["swapchain_right"]), out / "diff_source_vs_swapchain_right_heatmap.png")
    return metrics
