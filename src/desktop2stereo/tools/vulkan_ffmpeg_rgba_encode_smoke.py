"""Exercise native Vulkan Compute RGBA -> NV12 -> Vulkan Video encode."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg-bin", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="run for this duration at --fps; overrides --frames when positive",
    )
    args = parser.parse_args()
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("dimensions must be positive even values")
    if args.fps < 1 or args.frames < 1 or args.duration_seconds < 0:
        parser.error("fps and frames must be positive; duration-seconds cannot be negative")
    frame_count = max(1, int(round(args.duration_seconds * args.fps))) if args.duration_seconds > 0 else args.frames
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
        fps=args.fps,
        target_bitrate=40_000_000,
        peak_bitrate=46_000_000,
    )
    importer = CudaVulkanImageImporter()
    frame = None
    try:
        tensor = torch.empty(
            (args.height, args.width, 4), dtype=torch.uint8, device="cuda"
        )
        tensor[..., 3] = 255
        packet_count = 0
        packet_bytes = 0
        started = time.monotonic()
        for timestamp in range(frame_count):
            frame = encoder.acquire_rgba_frame()
            # Change the luma/chroma content each frame so the loop exercises
            # repeated CUDA writes, slot reuse, conversion, encode and drain.
            tensor[..., 0] = (32 + timestamp) % 256
            tensor[..., 1] = (96 + timestamp * 3) % 256
            tensor[..., 2] = (192 + timestamp * 5) % 256
            ready_value = importer.write_ffmpeg_rgba_frame(tensor, frame)
            encoder.encode_rgba_frame(ready_value=ready_value, timestamp=timestamp)
            frame = None
            while True:
                packet = encoder.read_packet()
                if not packet:
                    break
                packet_count += 1
                packet_bytes += len(packet)
        encoder.flush()
        while True:
            packet = encoder.read_packet()
            if not packet:
                break
            packet_count += 1
            packet_bytes += len(packet)
        if packet_count < 1 or packet_bytes < 1:
            raise RuntimeError("Vulkan Video accepted no encoded packet")
        elapsed = max(0.001, time.monotonic() - started)
        print(
            "vulkan_ffmpeg_rgba_encode_smoke: PASS "
            f"size={args.width}x{args.height} frames={frame_count} "
            f"elapsed={elapsed:.3f}s measured_fps={frame_count / elapsed:.2f} "
            f"packets={packet_count} packet_bytes={packet_bytes}"
        )
        return 0
    finally:
        # Preserve the native conversion/driver error; encoder.close() performs
        # the bounded teardown when a submission has already failed.
        importer.close()
        encoder.close()


if __name__ == "__main__":
    raise SystemExit(main())
