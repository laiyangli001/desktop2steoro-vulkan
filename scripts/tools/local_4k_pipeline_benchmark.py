from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(SRC)

import glfw
import torch

from capture.factory import create_capture_runner, create_capture_source, normalize_config
from utils.settings import load_settings
from capture.preprocess import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime
from capture.types import CaptureConfig
from stereo_runtime.adapter import StereoRuntimeConfig
from stereo_runtime.runtime import StereoRuntime
from viewer.viewer import StereoWindow

OUT_DIR = ROOT / "outputs" / "visual_regression"


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "fps_from_mean": 0.0}
    ordered = sorted(float(v) for v in values)
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms": round(float(mean), 4),
        "median_ms": round(float(statistics.median(values)), 4),
        "min_ms": round(float(min(values)), 4),
        "max_ms": round(float(max(values)), 4),
        "p90_ms": round(percentile(ordered, 0.90), 4),
        "p99_ms": round(percentile(ordered, 0.99), 4),
        "fps_from_mean": round(1000.0 / mean, 3) if mean > 0 else 0.0,
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def display_mode_for_output(output_format: str) -> str:
    return {
        "half_sbs": "Half-SBS",
        "full_sbs": "Full-SBS",
    }[output_format]


def collect_capture_frames(config: CaptureConfig, *, frames: int, timeout_s: float, warmup_frames: int = 0) -> tuple[list[tuple], dict]:
    runner = create_capture_runner(config)
    samples: "queue.Queue[tuple]" = queue.Queue(maxsize=max(1, frames * 2))
    errors: "queue.Queue[str]" = queue.Queue()
    done = threading.Event()

    def on_frame(frame_raw, size, capture_start_time):
        now = time.perf_counter()
        try:
            samples.put_nowait((frame_raw, size, capture_start_time, now))
        except queue.Full:
            pass
        if samples.qsize() >= frames:
            done.set()
            stop = getattr(runner, "stop", None)
            if callable(stop):
                stop()

    def on_error(exc):
        errors.put(f"{type(exc).__name__}: {exc}")
        done.set()

    # Polling capture sources are easier and avoid event backend keyboard side effects.
    if runner.__class__.__name__ == "PollingCaptureRunner":
        source = create_capture_source(config)
        try:
            for _ in range(frames):
                start = time.perf_counter()
                frame_raw, size = source.grab()
                samples.put((frame_raw, size, start, time.perf_counter()))
        finally:
            stop = getattr(source, "stop", None)
            if callable(stop):
                stop()
    else:
        shutdown = threading.Event()
        thread = threading.Thread(
            target=runner.run,
            kwargs={
                "shutdown_event": shutdown,
                "on_frame": on_frame,
                "on_error": on_error,
                "on_closed": done.set,
            },
            daemon=True,
        )
        thread.start()
        done.wait(timeout=timeout_s)
        shutdown.set()
        stop = getattr(runner, "stop", None)
        if callable(stop):
            stop()
        thread.join(timeout=2.0)

    captured = []
    while not samples.empty() and len(captured) < frames:
        captured.append(samples.get())
    if len(captured) < frames and not errors.empty():
        raise RuntimeError(errors.get())
    if not captured:
        raise RuntimeError("capture produced no frames")

    measure_start = min(max(0, warmup_frames), len(captured))
    measured = captured[measure_start:]
    intervals_ms = [
        (measured[i][3] - measured[i - 1][3]) * 1000.0
        for i in range(1, len(measured))
    ]
    capture_ms = [(end - start) * 1000.0 for _, _, start, end in measured]
    report = {
        "total_frames": len(captured),
        "warmup_frames": measure_start,
        "measured_frames": len(measured),
        "capture_call": stats(capture_ms),
        "arrival_interval": stats(intervals_ms),
        "fps_from_arrival_interval": round(1000.0 / statistics.fmean(intervals_ms), 3) if intervals_ms else 0.0,
        "runner": runner.__class__.__name__,
    }
    return captured, report


def make_runtime_config(args: argparse.Namespace, output_format: str) -> StereoRuntimeConfig:
    settings = load_settings(args.settings) if args.settings else {}
    return StereoRuntimeConfig(
        model_id=args.model_id or settings.get("Depth Model") or "Depth-Anything-V2-Small",
        cache_dir=args.cache_dir or Path(settings.get("Cache Path", args.settings.parent / "models" if args.settings else ROOT / "models")),
        mode=args.mode or "movie",
        stereo_quality=args.stereo_quality or str(settings.get("Stereo Quality", settings.get("Synthetic View", "quality_4k"))),
        output_format=output_format,
        depth_backend=args.depth_backend or ("tensorrt_native" if settings.get("TensorRT", False) else "onnx_cuda" if settings.get("ONNX", False) else "pytorch_cuda"),
        device=args.device,
        build_trt_engine=False,
        force_rebuild_trt=False,
        use_cuda_graph=args.cuda_graph,
        temporal=args.temporal,
        layers=args.layers if args.layers is not None else int(settings.get("Layers", 2)),
        hole_fill=args.hole_fill or str(settings.get("Hole Fill", "edge_aware")),
        depth_strength=args.depth_strength if args.depth_strength is not None else float(settings.get("Depth Strength", 2.0)),
        convergence=args.convergence if args.convergence is not None else float(settings.get("Convergence", 0.0)),
        fused=not args.disable_fused,
    )


def create_window(width: int, height: int, output_format: str, args: argparse.Namespace) -> StereoWindow:
    return StereoWindow(
        capture_mode=args.capture_mode,
        monitor_index=args.monitor_index,
        display_mode=display_mode_for_output(output_format),
        fill_16_9=True,
        show_fps=False,
        use_3d=False,
        fix_aspect=False,
        stream_mode=None,
        specify_display=False,
        frame_size=(width, height),
        use_cuda=True,
        cuda_device_id=args.cuda_device,
        local_vsync=args.vsync,
    )


def process_frames(
    captured_frames: list[tuple],
    *,
    output_format: str,
    args: argparse.Namespace,
) -> dict:
    runtime = StereoRuntime(make_runtime_config(args, output_format), collect_memory_stats=False)
    runtime.load()
    window = None
    rows = []
    first_result = None

    try:
        for idx, (frame_raw, size, _capture_start, capture_end) in enumerate(captured_frames):
            recording = idx >= args.warmup_frames
            timings = {"index": idx}

            start = time.perf_counter()
            frame_rgb = capture_frame_to_rgb(
                frame_raw,
                size,
                device=args.device,
                use_torch=True,
                output="tensor",
            )
            sync_cuda()
            timings["capture_preprocess_ms"] = ms_since(start)

            start = time.perf_counter()
            runtime_rgb = prepare_rgb_for_stereo_runtime(frame_rgb, device=args.device)
            sync_cuda()
            timings["prepare_runtime_rgb_ms"] = ms_since(start)

            start = time.perf_counter()
            result = runtime.process_rgb_frame(runtime_rgb)
            sync_cuda()
            timings["runtime_total_wall_ms"] = ms_since(start)
            for key, value in result.timing.items():
                timings[key] = float(value)

            if window is None:
                frame = result.sbs.detach()
                if frame.ndim == 4:
                    frame = frame[0]
                if frame.ndim == 3 and frame.shape[0] in (3, 4):
                    out_h, out_w = int(frame.shape[1]), int(frame.shape[2])
                else:
                    out_h, out_w = int(frame.shape[0]), int(frame.shape[1])
                window = create_window(out_w, out_h, output_format, args)

            start = time.perf_counter()
            window.update_runtime_frame(result)
            sync_cuda()
            timings["viewer_update_cuda_gl_ms"] = ms_since(start)

            start = time.perf_counter()
            window.render()
            sync_cuda()
            timings["viewer_render_ms"] = ms_since(start)

            start = time.perf_counter()
            glfw.swap_buffers(window.window)
            timings["swap_buffers_ms"] = ms_since(start)
            glfw.poll_events()

            timings["end_to_end_after_capture_ms"] = (
                timings["capture_preprocess_ms"]
                + timings["prepare_runtime_rgb_ms"]
                + timings["runtime_total_wall_ms"]
                + timings["viewer_update_cuda_gl_ms"]
                + timings["viewer_render_ms"]
                + timings["swap_buffers_ms"]
            )
            if recording:
                rows.append(timings)
            first_result = result if first_result is None else first_result
    finally:
        try:
            runtime.close()
        except Exception:
            pass
        if window is not None:
            try:
                window.cleanup_cuda()
            except Exception:
                pass
            try:
                glfw.destroy_window(window.window)
            except Exception:
                pass
            glfw.terminate()

    stage_keys = sorted({key for row in rows for key in row if key != "index"})
    stage_report = {key: stats([row[key] for row in rows if key in row]) for key in stage_keys}
    bottlenecks = sorted(
        (
            {"stage": key, "mean_ms": value["mean_ms"], "fps_from_mean": value["fps_from_mean"]}
            for key, value in stage_report.items()
            if key.endswith("_ms") or key == "total_ms"
        ),
        key=lambda item: item["mean_ms"],
        reverse=True,
    )
    provider_info = first_result.provider_info if first_result is not None else {}
    debug_info = first_result.debug_info if first_result is not None else {}
    return {
        "output_format": output_format,
        "warmup_frames": int(args.warmup_frames),
        "measured_frames": len(rows),
        "provider_info": provider_info,
        "debug_info": {
            key: value for key, value in debug_info.items()
            if isinstance(value, (str, int, float, bool))
        },
        "stages": stage_report,
        "bottlenecks_by_mean_ms": bottlenecks[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real local 4K capture -> stereo_runtime -> CUDA/GL output pipeline.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--capture-timeout", type=float, default=10.0)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--output-format", action="append", choices=["half_sbs", "full_sbs"], default=None)
    parser.add_argument("--output-resolution", type=str, default="3840x2160")
    parser.add_argument("--capture-mode", default=None)
    parser.add_argument("--monitor-index", type=int, default=None)
    parser.add_argument("--capture-tool", default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--settings", type=Path, default=SRC / "settings.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--depth-backend", default=None, choices=["auto", "tensorrt_native", "onnx_cuda", "pytorch_cuda"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--mode", default=None, choices=["auto", "movie", "game", "image", "debug"])
    parser.add_argument("--stereo-quality", default=None, choices=["fast", "fast_plus", "quality_4k", "hq_4k"])
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--hole-fill", default=None, choices=["none", "fast", "edge_aware"])
    parser.add_argument("--depth-strength", type=float, default=None)
    parser.add_argument("--convergence", type=float, default=None)
    parser.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--disable-fused", action="store_true")
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "local_4k_pipeline_benchmark_report.json")
    return parser.parse_args()


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"resolution must be WIDTHxHEIGHT, got {value!r}")
    return int(parts[0]), int(parts[1])


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    settings = load_settings(args.settings) if args.settings else {}
    args.capture_mode = args.capture_mode or str(settings.get("Capture Mode", "Monitor"))
    args.monitor_index = args.monitor_index if args.monitor_index is not None else int(settings.get("Monitor Index", 1))
    args.capture_tool = args.capture_tool or settings.get("Capture Tool")
    if args.fps is None:
        target_fps = int(settings.get("Target FPS", 0) or 0)
        args.fps = target_fps if 1 <= target_fps <= 240 else 60
    output_formats = args.output_format or ["half_sbs", "full_sbs"]
    output_w, output_h = parse_resolution(args.output_resolution)
    capture_config = CaptureConfig(
        output_resolution=(output_w, output_h),
        fps=args.fps,
        capture_mode=args.capture_mode,
        monitor_index=args.monitor_index,
        capture_tool=args.capture_tool,
        os_name="Windows" if os.name == "nt" else None,
    )

    effective_capture_config = normalize_config(capture_config)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captured, capture_report = collect_capture_frames(
        capture_config,
        frames=args.frames + args.warmup_frames,
        timeout_s=args.capture_timeout,
        warmup_frames=args.warmup_frames,
    )
    if args.capture_only:
        report = {
            "capture_config": {
                "output_resolution": [output_w, output_h],
                "fps": effective_capture_config.fps,
                "capture_mode": effective_capture_config.capture_mode,
                "monitor_index": effective_capture_config.monitor_index,
                "capture_tool": effective_capture_config.capture_tool,
            },
            "capture": capture_report,
            "warmup_frames": int(args.warmup_frames),
            "measured_frames": int(args.frames),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    cases = [
        process_frames(captured, output_format=fmt, args=args)
        for fmt in output_formats
    ]
    report = {
        "capture_config": {
            "output_resolution": [output_w, output_h],
            "fps": args.fps,
            "capture_mode": args.capture_mode,
            "monitor_index": args.monitor_index,
            "capture_tool": args.capture_tool,
        },
        "capture": capture_report,
        "warmup_frames": int(args.warmup_frames),
        "measured_frames": int(args.frames),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())











