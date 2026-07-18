from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workspace-gb", type=int, default=4)
    args = parser.parse_args()

    from stereo_runtime.depth_onnx_provider import default_distill_base_onnx_path
    from stereo_runtime.depth_trt_native_provider import build_native_tensorrt_engine, default_distill_base_native_trt_path

    onnx_path = Path(args.onnx) if args.onnx else default_distill_base_onnx_path(args.cache_dir)
    engine_path = Path(args.engine) if args.engine else default_distill_base_native_trt_path(args.cache_dir)

    print(f"[native-trt] onnx={onnx_path}", flush=True)
    print(f"[native-trt] engine={engine_path}", flush=True)
    print(f"[native-trt] force={args.force} workspace_gb={args.workspace_gb}", flush=True)
    start = time.perf_counter()
    built = build_native_tensorrt_engine(
        onnx_path,
        engine_path,
        workspace_gb=args.workspace_gb,
        force=args.force,
    )
    elapsed = time.perf_counter() - start
    print(f"[native-trt] ready={built}", flush=True)
    print(f"[native-trt] elapsed_sec={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
