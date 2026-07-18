from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def make_demo_frame(torch, width: int, height: int, device):
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    bg = torch.stack(
        [
            0.18 + 0.42 * xx,
            0.16 + 0.48 * yy,
            0.28 + 0.18 * torch.sin(xx * 10.0),
        ],
        dim=0,
    )

    circle = (((xx - 0.34) ** 2 / 0.055) + ((yy - 0.45) ** 2 / 0.095) < 1.0).float()
    panel = ((xx > 0.58) & (xx < 0.86) & (yy > 0.18) & (yy < 0.78)).float()
    bars = (((torch.sin(xx * 80.0) > 0.55) & (yy > 0.08) & (yy < 0.92))).float() * 0.35

    rgb = bg
    rgb = rgb * (1.0 - circle) + torch.tensor([0.95, 0.42, 0.12], device=device).view(3, 1, 1) * circle
    rgb = rgb * (1.0 - panel) + torch.tensor([0.08, 0.72, 0.74], device=device).view(3, 1, 1) * panel
    rgb = (rgb + bars).clamp(0, 1)

    depth = (0.18 + 0.42 * (1.0 - yy) + 0.14 * torch.sin(xx * 5.0)).clamp(0, 1)
    depth = depth * (1.0 - circle) + 0.92 * circle
    depth = depth * (1.0 - panel) + 0.68 * panel
    return rgb.unsqueeze(0).float(), depth.unsqueeze(0).unsqueeze(0).float()


def main() -> None:
    print("[1/5] importing torch ...", flush=True)
    import torch

    print("[2/5] importing stereo_runtime ...", flush=True)
    from stereo_runtime.io import save_depth, save_rgb
    from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)

    width, height = 1280, 720
    out_dir = ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/5] creating demo frame {width}x{height} ...", flush=True)
    rgb, depth = make_demo_frame(torch, width, height, device)
    save_rgb(rgb.cpu(), out_dir / "input_rgb.png")
    save_depth(depth.cpu(), out_dir / "input_depth.png")

    configs = [
        StereoConfig(backend="fast", output_format="half_sbs", debug_output=True, depth_strength=3.0),
        StereoConfig(backend="quality_4k", layers=2, output_format="half_sbs", debug_output=True, depth_strength=3.0),
        StereoConfig(backend="quality_4k", layers=2, output_format="full_sbs", debug_output=True, depth_strength=3.0),
        StereoConfig(backend="hq_4k", layers=3, output_format="half_sbs", debug_output=True, depth_strength=3.0),
    ]

    print("[4/5] synthesizing outputs ...", flush=True)
    with torch.inference_mode():
        for config in configs:
            result = synthesize_stereo(rgb, depth, config)
            suffix = f"{config.backend}_{config.output_format}"
            save_rgb(result.left_eye.cpu(), out_dir / f"{suffix}_left.png")
            save_rgb(result.right_eye.cpu(), out_dir / f"{suffix}_right.png")
            save_rgb(result.sbs.cpu(), out_dir / f"{suffix}.png")
            mask = result.debug_info.get("occlusion_mask")
            if mask is not None:
                save_depth(mask.cpu(), out_dir / f"{suffix}_occlusion_mask.png")
            print(f"  wrote {suffix}", flush=True)

    print(f"[5/5] demo images written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
