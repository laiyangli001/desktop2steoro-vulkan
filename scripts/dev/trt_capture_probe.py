from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from stereo_runtime.providers.nvidia.tensorrt_native import (
    NativeTensorRtEngine,
    default_native_tensorrt_engine_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal TensorRT enqueue/CUDA graph capture probe.")
    parser.add_argument("--engine", type=Path, default=default_native_tensorrt_engine_path())
    args = parser.parse_args()

    print(f"[probe] engine={args.engine}", flush=True)
    engine = NativeTensorRtEngine(args.engine, device="cuda", dtype=torch.float16)
    input_shape = engine.input_shape
    x = torch.empty(input_shape, device="cuda", dtype=torch.float16).normal_()
    torch.cuda.synchronize()
    print(f"[probe] input_shape={input_shape} dtype=fp16", flush=True)

    y = engine(x, synchronize=True)
    print(f"[probe] enqueue ok output_shape={tuple(y.shape)} dtype={y.dtype}", flush=True)

    engine.clear_graph()
    engine.capture_graph(input_shape)
    print("[probe] cuda graph capture ok", flush=True)

    y = engine.run_graph(x)
    torch.cuda.synchronize()
    print(f"[probe] graph replay ok output_shape={tuple(y.shape)} dtype={y.dtype}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
