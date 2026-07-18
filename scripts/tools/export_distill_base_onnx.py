from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.model_artifacts import artifact_paths_for_model
from stereo_runtime.onnx_export import (
    choose_export_dtype,
    export_depth_model_onnx,
    load_model_for_dtype,
    probe_model_dtype,
)


def default_output_path(model_id: str, dtype_name: str, height: int, width: int) -> Path:
    return artifact_paths_for_model(
        model_id,
        cache_dir=ROOT / "models",
        export_height=height,
        export_width=width,
    ).onnx_path_for_dtype(dtype_name)


def main() -> None:
    print("[1/5] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--model-id", default="lc700x/Distill-Any-Depth-Base-hf")
    parser.add_argument("--model-name", default="Distill-Any-Depth-Base")
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--no-force-download", action="store_true")
    args = parser.parse_args()

    print("[2/5] resolving output paths ...", flush=True)
    import torch

    model_id = args.model_id
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    _, dtype_name, dtype_reason = choose_export_dtype(model_id, device, args.dtype)
    output_path = Path(args.output) if args.output else default_output_path(model_id, dtype_name, args.height, args.width)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_path.parent.parent

    print(f"[info] model: {args.model_name}", flush=True)
    print(f"[info] model id: {model_id}", flush=True)
    print(f"[info] onnx input: 1x3x{args.height}x{args.width}", flush=True)
    print(f"[info] dtype request: {args.dtype}", flush=True)
    print(f"[info] initial dtype: {dtype_name}", flush=True)
    print(f"[info] dtype reason: {dtype_reason}", flush=True)
    print(f"[info] device: {device}", flush=True)
    print(f"[info] cache dir: {cache_dir}", flush=True)
    print(f"[info] output: {output_path}", flush=True)
    print(f"[info] force download: {not args.no_force_download}", flush=True)

    print("[3/5] loading/probing/exporting model ...", flush=True)
    result = export_depth_model_onnx(
        model_id=model_id,
        output_path=output_path,
        cache_dir=cache_dir,
        device=str(device),
        height=args.height,
        width=args.width,
        dtype=args.dtype,
        force_download=not args.no_force_download,
    )

    print("[4/5] export complete", flush=True)
    print(f"[info] final dtype: {result.dtype_name}", flush=True)
    print(f"[info] final dtype reason: {result.dtype_reason}", flush=True)
    print(f"[info] probe: {result.probe_reason}", flush=True)
    print(f"[info] output: {result.output_path}", flush=True)

    print("[5/5] ONNX written", flush=True)
    print(f"[info] size: {result.size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
