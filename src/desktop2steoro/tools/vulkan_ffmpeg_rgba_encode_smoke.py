"""Exercise native Vulkan Compute RGBA -> NV12 -> Vulkan Video encode."""

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
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("dimensions must be positive even values")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(args.ffmpeg_bin).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch

    from streaming.vulkan_bridge import VulkanNativeBridge
    from viewer.cuda_vulkan_interop import CudaVulkanImageImporter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    bridge = VulkanNativeBridge.load()
    encoder = bridge.create_encoder(
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
        tensor = torch.empty(
            (args.height, args.width, 4), dtype=torch.uint8, device="cuda"
        )
        tensor[..., 0] = 32
        tensor[..., 1] = 96
        tensor[..., 2] = 192
        tensor[..., 3] = 255
        ready_value = importer.write_ffmpeg_rgba_frame(tensor, frame)
        encoder.encode_rgba_frame(ready_value=ready_value, timestamp=0)
        frame = None
        packet = encoder.read_packet()
        if packet is None:
            encoder.flush()
            packet = encoder.read_packet()
        if not packet:
            raise RuntimeError("Vulkan Video accepted no encoded packet")
        print(
            "vulkan_ffmpeg_rgba_encode_smoke: PASS "
            f"size={args.width}x{args.height} ready={ready_value} "
            f"packet_bytes={len(packet)}"
        )
        return 0
    finally:
        if frame is not None:
            raise RuntimeError("encode smoke exited before native conversion")
        importer.close()
        encoder.close()


if __name__ == "__main__":
    raise SystemExit(main())
