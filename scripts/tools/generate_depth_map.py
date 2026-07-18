from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def colorize_depth(depth):
    import torch

    from stereo_runtime.output import ensure_b1hw

    depth = ensure_b1hw(depth).float().clamp(0, 1)
    near = depth
    mid = (1.0 - (depth - 0.5).abs() * 2.0).clamp(0, 1)
    far = 1.0 - depth
    return torch.cat([near, mid, far], dim=1).clamp(0, 1)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "depth"


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return safe_name(name), path
    path = Path(value)
    return safe_name(path.stem), value


def main() -> None:
    print("[1/6] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument(
        "--provider",
        action="append",
        choices=["distill_base_518", "distill_base_nvidia", "luma"],
        default=None,
        help="depth provider to run; repeat to compare multiple providers",
    )
    parser.add_argument(
        "--reference-depth",
        action="append",
        default=[],
        help="optional reference depth image; repeat or use name=path",
    )
    parser.add_argument("--out-dir", default="outputs/depth_compare")
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-cache-dir", default=None)
    parser.add_argument("--depth-onnx", default=None)
    parser.add_argument("--no-pytorch-fallback", action="store_true")
    parser.add_argument("--require-tensorrt", action="store_true")
    parser.add_argument("--depth-local-only", action="store_true")
    parser.add_argument("--depth-force-download", action="store_true")
    args = parser.parse_args()

    print("[2/6] importing torch ...", flush=True)
    import torch

    print("[3/6] importing stereo_runtime ...", flush=True)
    from stereo_runtime.auto_depth import estimate_luma_depth
    from stereo_runtime.depth_trt_provider import estimate_distill_any_depth_base_518_nvidia
    from stereo_runtime.depth_provider import estimate_distill_any_depth_base_518
    from stereo_runtime.io import load_depth, load_rgb, save_depth, save_rgb
    from stereo_runtime.output import match_depth
    from stereo_runtime.report import absdiff, depth_comparison_metrics, depth_metrics, make_contact_sheet, write_json

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    out_dir = Path(args.out_dir)
    providers = args.provider or ["distill_base_518"]
    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
    print(f"[info] depth providers: {', '.join(providers)}", flush=True)

    print("[4/6] loading RGB ...", flush=True)
    rgb = load_rgb(args.rgb, device=device)

    print("[5/6] estimating depth ...", flush=True)
    with torch.inference_mode():
        save_rgb(rgb.cpu(), out_dir / "input_rgb.png")
        generated_depths = {}
        sheet_items = [rgb.cpu()]
        report = {
            "rgb": str(args.rgb),
            "depth_providers": {},
            "outputs": {},
            "summary": {"primary_provider": providers[0], "providers": {}, "comparisons": {}},
            "comparisons": {},
        }

        for provider in providers:
            print(f"  running provider: {provider}", flush=True)
            if provider == "distill_base_nvidia":
                print("  model id: lc700x/Distill-Any-Depth-Base-hf", flush=True)
                print("  backend priority: tensorrt -> onnx_cuda_iobinding -> pytorch_cuda", flush=True)
                depth, provider_info = estimate_distill_any_depth_base_518_nvidia(
                    rgb,
                    device=device,
                    cache_dir=args.depth_cache_dir,
                    onnx_path=args.depth_onnx,
                    allow_pytorch_fallback=not args.no_pytorch_fallback,
                    require_tensorrt=args.require_tensorrt,
                    local_files_only=args.depth_local_only,
                    force_download=args.depth_force_download,
                )
                provider_report = provider_info.to_report()
            elif provider == "distill_base_518":
                print("  model id: lc700x/Distill-Any-Depth-Base-hf", flush=True)
                depth, provider_info = estimate_distill_any_depth_base_518(
                    rgb,
                    device=device,
                    cache_dir=args.depth_cache_dir,
                    local_files_only=args.depth_local_only,
                    force_download=args.depth_force_download,
                )
                provider_report = provider_info.to_report()
            elif provider == "luma":
                print("  warning: luma is pseudo-depth for workflow debugging only", flush=True)
                depth = estimate_luma_depth(rgb)
                provider_report = {
                    "provider": "luma_pseudo_depth",
                    "model_name": "none",
                    "model_id": "none",
                    "depth_resolution": "input",
                    "cache_dir": "none",
                    "load_mode": "local_math",
                }
            else:
                raise ValueError(f"unknown provider: {provider}")

            generated_depths[provider] = depth
            depth_file = f"{provider}_depth.png"
            color_file = f"{provider}_depth_color.png"
            save_depth(depth.cpu(), out_dir / depth_file)
            save_rgb(colorize_depth(depth).cpu(), out_dir / color_file)
            sheet_items.extend([depth.repeat(1, 3, 1, 1).cpu(), colorize_depth(depth).cpu()])
            metrics = depth_metrics(depth.cpu())
            report["depth_providers"][provider] = provider_report
            report["outputs"][provider] = {
                "depth": depth_file,
                "depth_color": color_file,
                "depth_shape": list(depth.shape),
                "metrics": metrics,
            }
            report["summary"]["providers"][provider] = metrics

        primary_name = providers[0]
        primary_depth = generated_depths[primary_name]
        primary_output = report["outputs"][primary_name]
        report["depth_provider"] = report["depth_providers"][primary_name]
        report["outputs"]["depth"] = primary_output["depth"]
        report["outputs"]["depth_color"] = primary_output["depth_color"]
        report["outputs"]["depth_shape"] = primary_output["depth_shape"]
        for provider, depth in generated_depths.items():
            if provider == primary_name:
                continue
            diff = absdiff(primary_depth.repeat(1, 3, 1, 1).cpu(), depth.repeat(1, 3, 1, 1).cpu())
            diff_file = f"{primary_name}_vs_{provider}_absdiff.png"
            save_rgb(diff, out_dir / diff_file)
            sheet_items.append(diff)
            comparison_key = f"{primary_name}_vs_{provider}"
            report["comparisons"][comparison_key] = depth_comparison_metrics(primary_depth.cpu(), depth.cpu())
            report["comparisons"][comparison_key]["absdiff"] = diff_file
            report["summary"]["comparisons"][comparison_key] = report["comparisons"][comparison_key]

        for reference_arg in args.reference_depth:
            reference_name, reference_path = parse_named_path(reference_arg)
            reference = load_depth(reference_path, device=device)
            reference = match_depth(reference, primary_depth.shape[-2], primary_depth.shape[-1])
            depth_rgb = primary_depth.repeat(1, 3, 1, 1)
            reference_rgb = reference.repeat(1, 3, 1, 1)
            diff = absdiff(reference_rgb.cpu(), depth_rgb.cpu())
            reference_file = f"reference_{reference_name}_matched.png"
            diff_file = f"reference_{reference_name}_vs_{primary_name}_absdiff.png"
            save_depth(reference.cpu(), out_dir / reference_file)
            save_rgb(diff, out_dir / diff_file)
            sheet_items.extend([reference_rgb.cpu(), diff])
            comparison_key = f"reference_{reference_name}_vs_{primary_name}"
            report["comparisons"][comparison_key] = depth_comparison_metrics(reference.cpu(), primary_depth.cpu())
            report["comparisons"][comparison_key]["reference_depth"] = str(reference_path)
            report["comparisons"][comparison_key]["reference_matched"] = reference_file
            report["comparisons"][comparison_key]["absdiff"] = diff_file
            report["summary"]["comparisons"][comparison_key] = report["comparisons"][comparison_key]

        contact = make_contact_sheet(sheet_items, columns=2)
        save_rgb(contact, out_dir / "depth_contact_sheet.png")
        write_json(report, out_dir / "depth_report.json")

    print(f"[6/6] depth output written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
