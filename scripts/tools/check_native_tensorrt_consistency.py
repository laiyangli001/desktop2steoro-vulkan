from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--out", default="outputs/native_tensorrt_consistency.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--trt-engine", default=None)
    args = parser.parse_args()

    import torch

    from stereo_runtime.depth_trt_native_provider import DistillAnyDepthBaseNativeTensorRt
    from stereo_runtime.io import load_rgb

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rgb = load_rgb(args.rgb, device=device)
    provider = DistillAnyDepthBaseNativeTensorRt(
        device=device,
        cache_dir=args.cache_dir,
        onnx_path=args.onnx,
        engine_path=args.trt_engine,
    )
    provider.load()

    with torch.inference_mode():
        first = provider.predict_profile(rgb)
        second = provider.predict_profile(rgb)
        diff = (first.depth - second.depth).abs()
        report = {
            "rgb": str(args.rgb),
            "provider": provider.info.to_report(),
            "first_profile": first.to_report(),
            "second_profile": second.to_report(),
            "depth_absdiff_mean": float(diff.mean().item()),
            "depth_absdiff_max": float(diff.max().item()),
            "depth_shape": list(first.depth.shape),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
