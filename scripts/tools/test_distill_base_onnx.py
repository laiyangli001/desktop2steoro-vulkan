from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def default_onnx_path() -> Path:
    return ROOT / "models" / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_294x518.onnx"


def normalize_depth(depth):
    import torch

    depth = depth.float()
    flat = depth.flatten(start_dim=1)
    amin = flat.amin(dim=-1).view(depth.shape[0], *([1] * (depth.ndim - 1)))
    amax = flat.amax(dim=-1).view(depth.shape[0], *([1] * (depth.ndim - 1)))
    return ((depth - amin) / (amax - amin).clamp_min(1e-6)).clamp(0, 1)


def make_model_input(rgb, height: int, width: int, device, dtype):
    import torch
    import torch.nn.functional as F

    tensor = F.interpolate(
        rgb.to(device).float().clamp(0, 1),
        size=(height, width),
        mode="bicubic" if device.type == "cuda" else "bilinear",
        align_corners=False,
        antialias=True if device.type == "cuda" else False,
    ).to(dtype)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--onnx", default=str(default_onnx_path()))
    parser.add_argument("--out-dir", default="outputs/onnx_distill_smoke")
    parser.add_argument("--cache-dir", default=str(ROOT / "models"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--width", type=int, default=518)
    args = parser.parse_args()

    print("[1/6] importing runtimes ...", flush=True)
    import numpy as np
    import torch
    import onnxruntime as ort
    from transformers import AutoModelForDepthEstimation

    from stereo_runtime.io import load_rgb, save_depth, save_rgb
    from stereo_runtime.report import absdiff, basic_image_metrics, write_json

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    out_dir = Path(args.out_dir)

    print("[2/6] preparing input ...", flush=True)
    rgb = load_rgb(args.rgb, device="cpu")
    model_input = make_model_input(rgb, args.height, args.width, device, dtype)
    ort_input = model_input.detach().cpu().numpy().astype(np.float16 if dtype == torch.float16 else np.float32)

    print("[3/6] running ONNX Runtime ...", flush=True)
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available and device.type == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_output = session.run(["predicted_depth"], {"pixel_values": ort_input})[0]
    onnx_depth = torch.from_numpy(onnx_output).float()
    if onnx_depth.ndim == 3:
        onnx_depth = onnx_depth.unsqueeze(1)
    onnx_depth = normalize_depth(onnx_depth)

    print("[4/6] running PyTorch reference ...", flush=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        "lc700x/Distill-Any-Depth-Base-hf",
        dtype=dtype,
        cache_dir=args.cache_dir,
        weights_only=True,
        local_files_only=True,
    ).to(device).eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda" and dtype == torch.float16):
        torch_output = model(pixel_values=model_input).predicted_depth
    torch_depth = torch_output.detach().float().cpu()
    if torch_depth.ndim == 3:
        torch_depth = torch_depth.unsqueeze(1)
    torch_depth = normalize_depth(torch_depth)

    print("[5/6] writing report ...", flush=True)
    diff = absdiff(torch_depth.repeat(1, 3, 1, 1), onnx_depth.repeat(1, 3, 1, 1))
    save_rgb(rgb, out_dir / "input_rgb.png")
    save_depth(onnx_depth, out_dir / "onnx_depth.png")
    save_depth(torch_depth, out_dir / "torch_depth.png")
    save_rgb(diff, out_dir / "torch_vs_onnx_absdiff.png")

    metrics = basic_image_metrics(torch_depth.repeat(1, 3, 1, 1), onnx_depth.repeat(1, 3, 1, 1))
    report = {
        "rgb": str(args.rgb),
        "onnx": str(onnx_path),
        "providers_available": available,
        "providers_used": session.get_providers(),
        "input_shape": list(ort_input.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "torch_output_shape": list(torch_output.shape),
        "metrics": metrics,
    }
    write_json(report, out_dir / "onnx_test_report.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[6/6] ONNX test written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
