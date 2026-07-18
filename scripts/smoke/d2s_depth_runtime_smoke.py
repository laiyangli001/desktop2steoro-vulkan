from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(SRC)

import numpy as np
import torch

from capture import capture_frame_to_rgb, prepare_rgb_for_depth_runtime
from stereo_runtime import DepthRuntime, runtime_config_from_d2s_settings
from stereo_runtime.depth_provider import DepthProfileResult, DepthProviderInfo
from stereo_runtime.output import ensure_bchw


class FakeDepthProvider:
    def __init__(self) -> None:
        self.load_count = 0
        self.predict_count = 0
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="fake-depth",
            model_id="fake/model",
            depth_resolution=518,
            cache_dir=".",
            load_mode="smoke",
            depth_backend="fake",
            runtime="d2s-depth-runtime-smoke",
        )

    def load(self) -> None:
        self.load_count += 1

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        self.predict_count += 1
        rgb = ensure_bchw(rgb, name="rgb")
        b, _, h, w = rgb.shape
        ramp = torch.linspace(0, 1, w, dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, w)
        depth = ramp.expand(b, 1, h, w).contiguous()
        return DepthProfileResult(depth=depth, preprocess_ms=0.1, model_ms=0.2, postprocess_ms=0.3)


def _make_bgra_frame(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    b = np.broadcast_to(x, (height, width))
    g = np.broadcast_to(y, (height, width))
    r = ((b.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
    a = np.full((height, width), 255, dtype=np.uint8)
    return np.stack([b, g, r, a], axis=-1)


def run_smoke(
    width: int,
    height: int,
    target_height: int,
    device: str,
    *,
    backend: str = "pytorch_cuda",
    real_provider: bool = False,
    model: str = "Distill-Any-Depth-Base",
) -> dict:
    settings = {
        "Depth Model": model,
        "TensorRT": backend in {"auto", "tensorrt_native"},
        "FP16": False,
        "Display Mode": "Half-SBS",
        "Run Mode": "Movie",
        "Depth Backend": backend,
    }
    config = runtime_config_from_d2s_settings(settings, cache_dir="./models", device=device)
    provider = None if real_provider else FakeDepthProvider()
    runtime = DepthRuntime(config, depth_provider=provider, collect_memory_stats=real_provider)
    depth_q: queue.Queue = queue.Queue(maxsize=1)

    frame_raw = _make_bgra_frame(width, height)
    capture_start_time = time.perf_counter()
    frame_rgb = capture_frame_to_rgb(frame_raw, target_height)
    runtime_rgb = prepare_rgb_for_depth_runtime(frame_rgb, device=device)
    depth_result = runtime.predict_depth_frame(runtime_rgb)
    depth_q.put((frame_rgb, depth_result.depth, capture_start_time))
    queued_rgb, queued_depth, queued_ts = depth_q.get_nowait()

    return {
        "mode": "d2s_depth_runtime",
        "raw_shape": list(frame_raw.shape),
        "frame_rgb_shape": list(queued_rgb.shape),
        "runtime_rgb_shape": list(runtime_rgb.shape),
        "runtime_rgb_dtype": str(runtime_rgb.dtype),
        "runtime_rgb_min": float(runtime_rgb.min().item()),
        "runtime_rgb_max": float(runtime_rgb.max().item()),
        "depth_shape": list(queued_depth.shape),
        "depth_dtype": str(queued_depth.dtype),
        "depth_min": float(queued_depth.min().item()),
        "depth_max": float(queued_depth.max().item()),
        "capture_timestamp_type": type(queued_ts).__name__,
        "queue_contract": "(frame_rgb, depth, capture_start_time)",
        "provider_load_count": getattr(provider, "load_count", None),
        "provider_predict_count": getattr(provider, "predict_count", None),
        "provider_info": depth_result.provider_info,
        "timing": depth_result.timing,
        "runtime_backend": config.depth_backend,
        "real_provider": real_provider,
        "resolved_model_id": config.resolved_model_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--target-height", type=int, default=2160)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="pytorch_cuda", choices=["auto", "pytorch_cuda", "onnx_cuda", "tensorrt_native"])
    parser.add_argument("--model", default="Distill-Any-Depth-Base")
    parser.add_argument("--real-provider", action="store_true")
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    report = run_smoke(
        args.width,
        args.height,
        args.target_height,
        args.device,
        backend=args.backend,
        real_provider=args.real_provider,
        model=args.model,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
