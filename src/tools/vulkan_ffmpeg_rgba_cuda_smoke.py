"""Smoke-test CUDA writes into an FFmpeg-owned Vulkan RGBA frame."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg-bin", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    args = parser.parse_args()

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(args.ffmpeg_bin).resolve()))
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "desktop2stereo")))

    import torch

    from streaming.vulkan_bridge import VulkanNativeBridge
    from viewer.cuda_vulkan_interop import CudaVulkanImageImporter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    bridge = os.environ.get("D2S_VULKAN_FFMPEG_BRIDGE")
    if not bridge:
        raise RuntimeError("D2S_VULKAN_FFMPEG_BRIDGE is not set")
    native_bridge = VulkanNativeBridge.load(Path(bridge))
    encoder = native_bridge.create_encoder(
        width=args.width,
        height=args.height,
        fps=30,
        target_bitrate=40_000_000,
        peak_bitrate=46_000_000,
    )
    importer = CudaVulkanImageImporter()
    frame = None
    try:
        frame = encoder.acquire_rgba_frame()
        tensor = torch.zeros(
            (args.height, args.width, 4), dtype=torch.uint8, device="cuda"
        )
        tensor[..., 0] = 32
        tensor[..., 1] = 96
        tensor[..., 2] = 192
        tensor[..., 3] = 255
        ready_value = importer.write_ffmpeg_rgba_frame(tensor, frame)
        torch.cuda.current_stream().synchronize()
        encoder.release_rgba_frame(ready_value)
        frame = None
        print(
            "vulkan_ffmpeg_rgba_cuda_smoke: PASS "
            f"size={args.width}x{args.height} ready={ready_value}"
        )
        return 0
    finally:
        if frame is not None:
            raise RuntimeError("RGBA smoke exited before producer timeline was returned")
        importer.close()
        encoder.close()


if __name__ == "__main__":
    raise SystemExit(main())
