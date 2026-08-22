"""Pixel-exact A/B comparison for the legacy and MIP screen sampling paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_DEFAULT_EYES = ("left", "right")
_MANIFEST_CANDIDATES = (
    "screen_sampling_runtime_manifest.json",
    "visual_regression_runtime_manifest.json",
)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _compare_pair(old: np.ndarray, new: np.ndarray) -> dict[str, float | int]:
    if old.shape != new.shape:
        raise ValueError(
            "screen sampling comparison requires identical image shapes: "
            f"old={old.shape} new={new.shape}"
        )
    diff = np.abs(old.astype(np.int16) - new.astype(np.int16))
    pixel_error = diff.max(axis=2)
    squared = diff.astype(np.float64) ** 2
    return {
        "width": int(old.shape[1]),
        "height": int(old.shape[0]),
        "exact_pixels": int(np.count_nonzero(pixel_error == 0)),
        "different_pixels": int(np.count_nonzero(pixel_error != 0)),
        "exact_pixel_ratio": float(np.mean(pixel_error == 0)),
        "different_pixel_ratio": float(np.mean(pixel_error != 0)),
        "mean_channel_error": float(np.mean(diff)),
        "rmse_channel_error": float(np.sqrt(np.mean(squared))),
        "max_channel_error": int(np.max(diff)),
        "pct_pixels_gt_1": float(np.mean(pixel_error > 1)),
        "pct_pixels_gt_5": float(np.mean(pixel_error > 5)),
    }


def _save_heatmap(old: np.ndarray, new: np.ndarray, path: Path) -> None:
    diff = np.abs(old.astype(np.int16) - new.astype(np.int16)).max(axis=2)
    scaled = np.clip(diff.astype(np.float32) * 8.0, 0.0, 255.0).astype(np.uint8)
    heatmap = np.empty((*scaled.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = scaled
    heatmap[..., 1] = np.minimum(scaled * 2, 255)
    heatmap[..., 2] = 255 - scaled
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heatmap, mode="RGB").save(path)


def _load_capture_manifest(root: Path) -> tuple[Path | None, dict[str, Any]]:
    for name in _MANIFEST_CANDIDATES:
        path = root / name
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, {}


def compare_screen_sampling_capture_dirs(
    legacy_dir: str | Path,
    mip_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    stage: str = "07_filament_screen",
    eyes: tuple[str, ...] = _DEFAULT_EYES,
    verify_source: bool = True,
) -> dict[str, Any]:
    """Compare real runtime PNGs captured with legacy and MIP sampling.

    The default stage is the camera-fixed Filament screen-only readback. It
    excludes OpenXR pose, projection, and controller composition effects.
    ``06_openxr_projection`` remains available for end-to-end diagnostics, and
    ``03_vulkan_output`` checks whether both runs used the same producer image.
    """
    if stage not in {
        "03_vulkan_output",
        "06_openxr_projection",
        "07_filament_screen",
    }:
        raise ValueError(
            "stage must be '03_vulkan_output', '06_openxr_projection', "
            "or '07_filament_screen'"
        )
    old_root = Path(legacy_dir)
    new_root = Path(mip_dir)
    output = (
        Path(output_dir)
        if output_dir is not None
        else new_root / "screen_sampling_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_status: dict[str, str] = {}
    manifest_paths: dict[str, str | None] = {}
    for label, root, expected in (
        ("legacy", old_root, "legacy"),
        ("mip", new_root, "mip"),
    ):
        manifest_path, manifest = _load_capture_manifest(root)
        manifest_paths[label] = str(manifest_path) if manifest_path else None
        if manifest_path is None:
            manifest_status[label] = "missing"
            continue
        actual = str(manifest.get("screen_sampling_mode", "")).strip().lower()
        if actual and actual != expected:
            raise ValueError(
                f"{label} capture manifest says screen_sampling_mode={actual!r}; "
                f"expected {expected!r}"
            )
        suffix = f":{manifest_path.name}"
        manifest_status[label] = (
            f"validated{suffix}" if actual else f"mode_missing{suffix}"
        )
    def compare_stage(stage_name: str) -> dict[str, dict[str, float | int]]:
        result_for_stage: dict[str, dict[str, float | int]] = {}
        for eye in eyes:
            old_path = old_root / f"{stage_name}_{eye}_eye.png"
            new_path = new_root / f"{stage_name}_{eye}_eye.png"
            if not old_path.is_file() or not new_path.is_file():
                missing = [
                    str(path) for path in (old_path, new_path) if not path.is_file()
                ]
                raise FileNotFoundError(
                    "missing screen sampling regression image(s): "
                    + ", ".join(missing)
                )
            old = _load_rgb(old_path)
            new = _load_rgb(new_path)
            metrics = _compare_pair(old, new)
            result_for_stage[eye] = metrics
            _save_heatmap(
                old, new, output / f"{stage_name}_{eye}_diff_heatmap.png"
            )
        return result_for_stage

    source_pairs = (
        compare_stage("03_vulkan_output")
        if verify_source and stage != "03_vulkan_output"
        else None
    )
    pairs = compare_stage(stage)

    result: dict[str, Any] = {
        "stage": stage,
        "legacy_dir": str(old_root.resolve()),
        "mip_dir": str(new_root.resolve()),
        "manifest_status": manifest_status,
        "manifest_paths": manifest_paths,
        "pairs": pairs,
        "mean_channel_error": float(
            np.mean([float(metrics["mean_channel_error"]) for metrics in pairs.values()])
        ),
        "different_pixel_ratio": float(
            np.mean([float(metrics["different_pixel_ratio"]) for metrics in pairs.values()])
        ),
    }
    if source_pairs is not None:
        result["source_verification"] = {
            "stage": "03_vulkan_output",
            "pairs": source_pairs,
            "mean_channel_error": float(
                np.mean(
                    [
                        float(metrics["mean_channel_error"])
                        for metrics in source_pairs.values()
                    ]
                )
            ),
            "different_pixel_ratio": float(
                np.mean(
                    [
                        float(metrics["different_pixel_ratio"])
                        for metrics in source_pairs.values()
                    ]
                )
            ),
        }
    (output / "screen_sampling_pixel_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
