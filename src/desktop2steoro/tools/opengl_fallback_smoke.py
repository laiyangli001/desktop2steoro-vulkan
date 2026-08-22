"""Probe the OpenGL fallback image path and reusable PBO/fence ring.

This tool deliberately does not start MediaMTX. It validates the image boundary
before a streaming session: headless context creation, RGBA8 texture, PBO/fence
slot reuse, and (when available) CUDA/HIP graphics interop. The output is a
single JSON object suitable for attaching to a platform validation report.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from streaming.opengl_stream_backend import OpenGLFallbackBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--pbo-count", type=int, default=3)
    parser.add_argument(
        "--require-gpu-interop",
        action="store_true",
        help="fail if CUDA/HIP OpenGL graphics interop is unavailable",
    )
    parser.add_argument(
        "--force-host",
        action="store_true",
        help="skip CUDA/HIP probe and exercise the PBO/fence host path",
    )
    return parser.parse_args()


def _gpu_probe(backend: OpenGLFallbackBackend, frames: int) -> dict[str, Any]:
    import torch

    if not bool(torch.cuda.is_available()):
        return {"tested": False, "reason": "CUDA/HIP tensor runtime unavailable"}
    if not backend.capabilities.cuda_gl_interop and not backend.capabilities.hip_gl_interop:
        return {"tested": False, "reason": backend.capabilities.detail}

    source = torch.zeros(
        (3, backend.height, backend.width), dtype=torch.uint8, device="cuda"
    )
    source[0, 0, 0] = 17
    started = time.perf_counter()
    output = None
    for _ in range(frames):
        output = backend.submit_cuda(source)
    torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - started, 1e-9)
    assert output is not None
    return {
        "tested": True,
        "path": "gpu-interop",
        "frames": frames,
        "elapsed_seconds": round(elapsed, 6),
        "fps": round(frames / elapsed, 3),
        "device": str(output.device),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "gpu_to_cpu": False,
        "zero_copy": backend.capabilities.zero_copy,
    }


def _host_probe(backend: OpenGLFallbackBackend, frames: int) -> dict[str, Any]:
    frame = np.zeros((backend.height, backend.width, 3), dtype=np.uint8)
    frame[0, 0] = [17, 23, 42]
    started = time.perf_counter()
    output = None
    for _ in range(frames):
        output = backend.submit_rgb(frame)
    elapsed = max(time.perf_counter() - started, 1e-9)
    assert output is not None
    return {
        "tested": True,
        "path": "host-upload",
        "frames": frames,
        "elapsed_seconds": round(elapsed, 6),
        "fps": round(frames / elapsed, 3),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "gpu_to_cpu": True,
        "zero_copy": False,
    }


def main() -> int:
    args = _parse_args()
    if args.width < 2 or args.height < 2 or args.frames < 1 or args.pbo_count < 2:
        raise SystemExit("width/height/frames must be positive and pbo-count must be >= 2")

    backend = OpenGLFallbackBackend(args.width, args.height, pbo_count=args.pbo_count)
    try:
        capabilities = backend.capabilities
        gpu = (
            {"tested": False, "reason": "forced by --force-host"}
            if args.force_host
            else _gpu_probe(backend, args.frames)
        )
        host = None if gpu["tested"] else _host_probe(backend, args.frames)
        if args.require_gpu_interop and not gpu["tested"]:
            print(json.dumps({"capabilities": capabilities.__dict__, "gpu": gpu}, ensure_ascii=False))
            return 2
        result = {
            "capabilities": capabilities.__dict__,
            "gpu_probe": gpu,
            "host_probe": host,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
