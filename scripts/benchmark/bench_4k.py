from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.output import OUTPUT_FORMAT_CHOICES


def make_synthetic_frame(torch, width: int, height: int, device):
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rgb = torch.stack([xx, yy, 0.5 + 0.5 * torch.sin(xx * 12.0)], dim=0).unsqueeze(0)
    depth = ((xx * 0.45 + yy * 0.25) + 0.3 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)).clamp(0, 1)
    return rgb.float(), depth.unsqueeze(0).unsqueeze(0).float()


def main() -> None:
    print("[1/5] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--backend", choices=["fast", "fast_plus", "quality_4k", "hq_4k"], default="quality_4k")
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        default="half_sbs",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-strength", type=float, default=2.0)
    parser.add_argument("--convergence", type=float, default=0.0)
    parser.add_argument("--ipd", type=float, default=0.064)
    parser.add_argument("--max-shift-ratio", type=float, default=0.05)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--temporal-strength", type=float, default=0.85)
    parser.add_argument("--auto-reset-temporal", action="store_true")
    parser.add_argument("--scene-reset-threshold", type=float, default=0.22)
    parser.add_argument("--foreground-scale", type=float, default=0.0)
    parser.add_argument("--depth-antialias-strength", type=float, default=0.0)
    parser.add_argument("--edge-dilation", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--cross-eyed", action="store_true")
    parser.add_argument("--anaglyph-method", choices=["red_cyan", "green_magenta", "amber_blue", "gray"], default="red_cyan")
    args = parser.parse_args()

    print("[2/5] importing torch ...", flush=True)
    import torch

    print("[3/5] importing stereo_runtime ...", flush=True)
    from stereo_runtime.metrics import BenchStats, read_peak_memory_mb, reset_peak_memory, timed
    from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device_name}", flush=True)

    device = torch.device(device_name)

    print(f"[4/5] creating synthetic {args.width}x{args.height} frame ...", flush=True)
    rgb, depth = make_synthetic_frame(torch, args.width, args.height, device)
    config = StereoConfig(
        backend=args.backend,
        layers=args.layers,
        output_format=args.output_format,
        depth_strength=args.depth_strength,
        convergence=args.convergence,
        ipd=args.ipd,
        max_shift_ratio=args.max_shift_ratio,
        temporal=args.temporal,
        temporal_strength=args.temporal_strength,
        auto_reset_temporal=args.auto_reset_temporal,
        scene_reset_threshold=args.scene_reset_threshold,
        foreground_scale=args.foreground_scale,
        depth_antialias_strength=args.depth_antialias_strength,
        edge_dilation=args.edge_dilation,
        edge_threshold=args.edge_threshold,
        cross_eyed=args.cross_eyed,
        anaglyph_method=args.anaglyph_method,
        debug_output=False,
    )
    reset_peak_memory()

    print(f"[5/5] running benchmark frames={args.frames} ...", flush=True)
    stats = BenchStats()
    with torch.inference_mode():
        synthesize_stereo(rgb, depth, config)
        with timed(stats, "synthesis_loop"):
            result = None
            for _ in range(args.frames):
                result = synthesize_stereo(rgb, depth, config)
    stats.peak_memory_mb = read_peak_memory_mb()
    stats.output_shape = tuple(result.sbs.shape) if result is not None else None

    avg_ms = stats.timings_ms["synthesis_loop"] / max(args.frames, 1)
    fps = 1000.0 / avg_ms if avg_ms else 0.0
    print(f"backend={args.backend}")
    print(f"output_format={args.output_format}")
    print(f"input={args.width}x{args.height}")
    print(f"output_shape={stats.output_shape}")
    print(f"avg_ms={avg_ms:.3f}")
    print(f"fps={fps:.2f}")
    print(f"peak_memory_mb={stats.peak_memory_mb:.1f}")


if __name__ == "__main__":
    main()

