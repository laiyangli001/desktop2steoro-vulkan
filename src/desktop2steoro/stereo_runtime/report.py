from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from utils.cpu_warnings import describe_tensor, warn_cpu_operation, warn_cpu_transfer

from .layers import depth_edges
from .output import ensure_b1hw, ensure_bchw


def absdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = ensure_bchw(a, name="a").float()
    b = ensure_bchw(b, name="b").float()
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"channel mismatch: {a.shape[1]} vs {b.shape[1]}")
    return (a - b).abs().clamp(0, 1)


def basic_image_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = absdiff(a, b)
    warn_cpu_operation(
        "stereo_runtime.basic_image_metrics",
        "tensor stats .item() sync",
        detail=describe_tensor(diff),
        key="stereo_runtime_basic_image_metrics_cpu_stats",
    )
    mse = float((diff * diff).mean().item())
    mae = float(diff.mean().item())
    psnr = 99.0 if mse <= 1e-12 else float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())
    return {"mae": mae, "mse": mse, "psnr": psnr}


def depth_metrics(depth: torch.Tensor, bins: int = 16, edge_threshold: float = 0.04) -> dict[str, float | list[float]]:
    depth = ensure_b1hw(depth).float().clamp(0, 1)
    flat = depth.flatten()
    warn_cpu_transfer(
        "stereo_runtime.depth_metrics",
        "histogram flat.cpu()",
        detail=describe_tensor(flat),
        key="stereo_runtime_depth_metrics_histogram_cpu",
    )
    warn_cpu_operation(
        "stereo_runtime.depth_metrics",
        "tensor stats .item()/tolist() sync",
        detail=describe_tensor(depth),
        key="stereo_runtime_depth_metrics_cpu_stats",
    )
    histogram = torch.histc(flat.cpu(), bins=bins, min=0.0, max=1.0)
    histogram = histogram / histogram.sum().clamp_min(1.0)

    foreground = depth >= 0.65
    background = depth <= 0.35
    foreground_mean = float(depth[foreground].mean().item()) if foreground.any() else 0.0
    background_mean = float(depth[background].mean().item()) if background.any() else 0.0
    edges = depth_edges(depth, threshold=edge_threshold)

    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "foreground_ratio": float(foreground.float().mean().item()),
        "background_ratio": float(background.float().mean().item()),
        "foreground_mean": foreground_mean,
        "background_mean": background_mean,
        "foreground_background_separation": foreground_mean - background_mean,
        "edge_density": float(edges.mean().item()),
        "histogram": [float(x) for x in histogram.tolist()],
    }


def depth_comparison_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = ensure_b1hw(reference).float().clamp(0, 1)
    candidate = ensure_b1hw(candidate).float().clamp(0, 1)
    if candidate.shape[-2:] != reference.shape[-2:]:
        candidate = F.interpolate(candidate, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    warn_cpu_operation(
        "stereo_runtime.depth_comparison_metrics",
        "tensor stats .item() sync",
        detail=f"reference={describe_tensor(reference)} candidate={describe_tensor(candidate)}",
        key="stereo_runtime_depth_comparison_cpu_stats",
    )
    image_metrics = basic_image_metrics(reference.repeat(1, 3, 1, 1), candidate.repeat(1, 3, 1, 1))
    ref_edges = depth_edges(reference)
    cand_edges = depth_edges(candidate)
    edge_overlap = (ref_edges * cand_edges).sum() / (ref_edges.clamp_min(0) + cand_edges.clamp_min(0)).clamp(0, 1).sum().clamp_min(1.0)
    return {
        **image_metrics,
        "mean_bias": float((candidate - reference).mean().item()),
        "edge_overlap": float(edge_overlap.item()),
    }


def make_contact_sheet(images: list[torch.Tensor], columns: int = 2, pad: int = 8) -> torch.Tensor:
    if not images:
        raise ValueError("images must not be empty")
    tensors = [ensure_bchw(x, name="image").float() for x in images]
    if any(x.shape[0] != 1 for x in tensors):
        raise ValueError("contact sheet currently expects batch size 1")

    max_h = max(x.shape[-2] for x in tensors)
    max_w = max(x.shape[-1] for x in tensors)
    rows = (len(tensors) + columns - 1) // columns
    canvas = torch.zeros(1, tensors[0].shape[1], rows * max_h + (rows - 1) * pad, columns * max_w + (columns - 1) * pad)
    for idx, image in enumerate(tensors):
        row = idx // columns
        col = idx % columns
        y = row * (max_h + pad)
        x = col * (max_w + pad)
        h, w = image.shape[-2:]
        warn_cpu_transfer(
            "stereo_runtime.make_contact_sheet",
            "image.cpu() canvas copy",
            detail=describe_tensor(image),
            key="stereo_runtime_contact_sheet_image_cpu",
        )
        canvas[:, :, y : y + h, x : x + w] = image.cpu()
    return canvas


def make_labeled_contact_sheet(
    items: list[tuple[str, torch.Tensor]],
    columns: int = 2,
    pad: int = 8,
    label_height: int = 34,
) -> torch.Tensor:
    if not items:
        raise ValueError("items must not be empty")
    labels = [label for label, _ in items]
    tensors = [ensure_bchw(x, name="image").float() for _, x in items]
    if any(x.shape[0] != 1 for x in tensors):
        raise ValueError("contact sheet currently expects batch size 1")

    max_h = max(x.shape[-2] for x in tensors)
    max_w = max(x.shape[-1] for x in tensors)
    rows = (len(tensors) + columns - 1) // columns
    cell_h = label_height + max_h
    canvas_h = rows * cell_h + (rows - 1) * pad
    canvas_w = columns * max_w + (columns - 1) * pad
    canvas = torch.zeros(1, tensors[0].shape[1], canvas_h, canvas_w)
    for idx, image in enumerate(tensors):
        row = idx // columns
        col = idx % columns
        y = row * (cell_h + pad) + label_height
        x = col * (max_w + pad)
        h, w = image.shape[-2:]
        warn_cpu_transfer(
            "stereo_runtime.make_labeled_contact_sheet",
            "image.cpu() canvas copy",
            detail=describe_tensor(image),
            key="stereo_runtime_labeled_contact_sheet_image_cpu",
        )
        canvas[:, :, y : y + h, x : x + w] = image.cpu()
    return _draw_labels(canvas, labels, columns=columns, cell_width=max_w, cell_height=cell_h, pad=pad, label_height=label_height)


def _draw_labels(
    canvas: torch.Tensor,
    labels: list[str],
    columns: int,
    cell_width: int,
    cell_height: int,
    pad: int,
    label_height: int,
) -> torch.Tensor:
    from PIL import Image, ImageDraw, ImageFont

    label_tensor = canvas[0].clamp(0, 1).permute(1, 2, 0).mul(255).byte()
    warn_cpu_transfer(
        "stereo_runtime.draw_labels",
        ".byte().numpy() for PIL labels",
        detail=describe_tensor(label_tensor),
        key="stereo_runtime_draw_labels_cpu_numpy",
    )
    array = label_tensor.numpy()
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, label in enumerate(labels):
        row = idx // columns
        col = idx % columns
        x = col * (cell_width + pad)
        y = row * (cell_height + pad)
        draw.rectangle((x, y, x + cell_width - 1, y + label_height - 1), fill=(16, 16, 16))
        draw.text((x + 10, y + 9), label, fill=(255, 255, 255), font=font)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(image.height, image.width, 3).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def write_json(data: dict, path: str | Path) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
