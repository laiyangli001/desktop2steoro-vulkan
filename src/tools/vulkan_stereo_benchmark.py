from __future__ import annotations

import argparse
import json
import sys
import time
from array import array
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereo_runtime.vulkan_stereo_pass import VulkanStereoFusedParams, VulkanStereoFusedPass
from viewer.vulkan_context import VulkanContext
from viewer.vulkan_descriptors import VulkanStorageBuffer


def _float_bytes(value: float, count: int) -> bytes:
    return (array("f", [float(value)]) * int(count)).tobytes()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the fused Vulkan stereo compute pass")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.width < 1 or args.height < 1 or args.warmup < 0 or args.iterations < 1:
        raise SystemExit("width, height, iterations must be positive; warmup cannot be negative")
    pixels = int(args.width) * int(args.height)
    with VulkanContext.create() as context:
        with ExitStack() as stack:
            stereo_pass = stack.enter_context(
                VulkanStereoFusedPass(context, width=args.width, height=args.height)
            )
            buffers = {
                name: stack.enter_context(VulkanStorageBuffer(context, size))
                for name, size in stereo_pass.buffer_sizes.items()
            }
            buffers["rgb"].write_bytes(_float_bytes(0.25, pixels * 3))
            buffers["depth"].write_bytes(_float_bytes(0.5, pixels))
            params = VulkanStereoFusedParams(max_disparity_px=96.0)

            def dispatch(frame_id: int) -> None:
                stereo_pass.submit(
                    buffers["rgb"],
                    buffers["depth"],
                    buffers["left_eye"],
                    buffers["right_eye"],
                    buffers["occlusion_mask"],
                    params=params,
                    frame_id=frame_id,
                    config_version=1,
                )
                context.wait_idle()

            for frame_id in range(int(args.warmup)):
                dispatch(frame_id)
            samples = []
            for frame_id in range(int(args.iterations)):
                start = time.perf_counter()
                dispatch(frame_id + int(args.warmup))
                samples.append((time.perf_counter() - start) * 1000.0)

            mean_ms = sum(samples) / len(samples)
            result = {
                "backend": "vulkan_fused_stereo",
                "device": context.device_info.name,
                "vendor_id": context.device_info.vendor_id,
                "width": int(args.width),
                "height": int(args.height),
                "warmup": int(args.warmup),
                "iterations": int(args.iterations),
                "dispatch_submit_wait_ms_mean": mean_ms,
                "dispatch_submit_wait_ms_min": min(samples),
                "dispatch_submit_wait_ms_max": max(samples),
                "estimated_fps": 1000.0 / mean_ms if mean_ms > 0.0 else 0.0,
                "groups": list(stereo_pass.group_counts),
                "note": "CPU wall time around one Vulkan submission and device wait; upload excluded",
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
