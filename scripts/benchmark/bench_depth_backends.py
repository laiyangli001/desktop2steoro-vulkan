from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p90_ms": float(percentile(values, 0.9)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def make_backend_provider(backend: str, *, device, cache_dir, onnx_path, trt_engine, model_id, model_name, build_native_engine, force_rebuild_native):
    from stereo_runtime.depth_onnx_provider import DistillAnyDepthBaseOnnxCuda
    from stereo_runtime.depth_provider import DistillAnyDepthBase518
    from stereo_runtime.depth_trt_native_provider import DistillAnyDepthBaseNativeTensorRt
    from stereo_runtime.depth_trt_provider import DistillAnyDepthBaseTensorRtOrt

    if backend == "tensorrt_native":
        return DistillAnyDepthBaseNativeTensorRt(
            device=device,
            cache_dir=cache_dir,
            onnx_path=onnx_path,
            engine_path=trt_engine,
            model_id=model_id,
            model_name=model_name,
            build_engine=build_native_engine,
            force_rebuild=force_rebuild_native,
        )
    if backend == "tensorrt_native_graph":
        return DistillAnyDepthBaseNativeTensorRt(
            device=device,
            cache_dir=cache_dir,
            onnx_path=onnx_path,
            engine_path=trt_engine,
            model_id=model_id,
            model_name=model_name,
            build_engine=build_native_engine,
            force_rebuild=force_rebuild_native,
            use_cuda_graph=True,
        )
    if backend == "tensorrt":
        return DistillAnyDepthBaseTensorRtOrt(device=device, cache_dir=cache_dir, onnx_path=onnx_path, model_id=model_id, model_name=model_name)
    if backend == "onnx_cuda_iobinding":
        return DistillAnyDepthBaseOnnxCuda(device=device, cache_dir=cache_dir, onnx_path=onnx_path, model_id=model_id, model_name=model_name, use_iobinding=True)
    if backend == "onnx_cuda_dlpack":
        return DistillAnyDepthBaseOnnxCuda(device=device, cache_dir=cache_dir, onnx_path=onnx_path, model_id=model_id, model_name=model_name, use_iobinding=True, use_dlpack=True)
    if backend == "pytorch_cuda":
        return DistillAnyDepthBase518(device=device, cache_dir=cache_dir, local_files_only=True)
    if backend == "nvidia_chain":
        return DistillAnyDepthBaseTensorRtOrt(device=device, cache_dir=cache_dir, onnx_path=onnx_path, model_id=model_id, model_name=model_name)
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--backend", choices=["tensorrt_native_graph", "tensorrt_native", "tensorrt", "onnx_cuda_iobinding", "onnx_cuda_dlpack", "pytorch_cuda", "nvidia_chain"], action="append", default=None)
    parser.add_argument("--out-dir", default="outputs/depth_backend_bench")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--trt-engine", default=None)
    parser.add_argument("--model-id", default="lc700x/Distill-Any-Depth-Base-hf")
    parser.add_argument("--model-name", default="Distill-Any-Depth-Base")
    parser.add_argument("--build-native-engine", action="store_true")
    parser.add_argument("--force-rebuild-native", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--require-tensorrt", action="store_true")
    args = parser.parse_args()

    import torch

    from stereo_runtime.io import load_rgb
    from stereo_runtime.report import write_json

    backends = args.backend or ["tensorrt", "onnx_cuda_iobinding", "pytorch_cuda"]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rgb = load_rgb(args.rgb, device=device)
    out_dir = Path(args.out_dir)

    env = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import onnxruntime as ort

        env["onnxruntime"] = ort.__version__
        env["ort_available_providers"] = ort.get_available_providers()
    except Exception as exc:
        env["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"

    results = {"rgb": str(args.rgb), "environment": env, "backends": {}}

    for backend in backends:
        print(f"[bench] backend={backend}", flush=True)
        timings: list[float] = []
        warmup_timings: list[float] = []
        preprocess_timings: list[float] = []
        model_timings: list[float] = []
        postprocess_timings: list[float] = []
        warmup_profile_timings: list[dict[str, float]] = []
        profile_timings: list[dict[str, float]] = []
        setup_ms = 0.0
        status = "ok"
        error = None
        provider_report = None
        output_shape = None
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            setup_start = time.perf_counter()
            provider = make_backend_provider(
                backend,
                device=device,
                cache_dir=args.cache_dir,
                onnx_path=args.onnx,
                trt_engine=args.trt_engine,
                model_id=args.model_id,
                model_name=args.model_name,
                build_native_engine=args.build_native_engine,
                force_rebuild_native=args.force_rebuild_native,
            )
            if hasattr(provider, "load"):
                provider.load()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            setup_ms = (time.perf_counter() - setup_start) * 1000.0
            provider_report = provider.info.to_report()
            print(f"  setup_ms={setup_ms:.3f}", flush=True)

            for iteration in range(args.warmup + args.iters):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if hasattr(provider, "predict_profile"):
                    profile = provider.predict_profile(rgb)
                    depth = profile.depth
                    profile_report = profile.to_report()
                else:
                    depth = provider.predict(rgb)
                    profile_report = None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                provider_report = provider.info.to_report()
                output_shape = list(depth.shape)
                if iteration >= args.warmup:
                    timings.append(elapsed_ms)
                    if profile_report is not None:
                        preprocess_timings.append(profile_report["preprocess_ms"])
                        model_timings.append(profile_report["model_ms"])
                        postprocess_timings.append(profile_report["postprocess_ms"])
                        profile_timings.append(profile_report)
                else:
                    warmup_timings.append(elapsed_ms)
                    if profile_report is not None:
                        warmup_profile_timings.append(profile_report)
                print(f"  iter={iteration + 1} elapsed_ms={elapsed_ms:.3f}", flush=True)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(f"  failed: {error}", flush=True)

        results["backends"][backend] = {
            "status": status,
            "error": error,
            "provider": provider_report,
            "output_shape": output_shape,
            "setup_ms": setup_ms,
            "warmup": summarize_ms(warmup_timings),
            "timings": summarize_ms(timings),
            "profile": {
                "preprocess": summarize_ms(preprocess_timings),
                "model": summarize_ms(model_timings),
                "postprocess": summarize_ms(postprocess_timings),
            },
            "raw_ms": timings,
            "warmup_raw_ms": warmup_timings,
            "profile_raw_ms": profile_timings,
            "warmup_profile_raw_ms": warmup_profile_timings,
        }

    write_json(results, out_dir / "depth_backend_bench.json")
    print(f"[done] wrote {out_dir / 'depth_backend_bench.json'}", flush=True)


if __name__ == "__main__":
    main()
