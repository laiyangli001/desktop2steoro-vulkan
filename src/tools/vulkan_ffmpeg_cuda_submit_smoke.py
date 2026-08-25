"""Exercise the CUDA -> exported FFmpeg Vulkan NV12 -> Vulkan Video hand-off.

This is a diagnostic tool, not the streaming transport.  It proves that the
same GPU timeline semaphore which CUDA signals is consumed by FFmpeg Vulkan
Video, without downloading an RGB frame or feeding raw pixels through stdin.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_dll_search_path(directory: str | None) -> None:
    if directory and os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(directory).resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg-bin", required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--hevc", action="store_true")
    args = parser.parse_args()
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("NV12 dimensions must be positive even values")
    if args.frames < 1:
        parser.error("frames must be positive")

    _configure_dll_search_path(args.ffmpeg_bin)
    source_root = (Path(__file__).resolve().parents[1] / "desktop2stereo")
    sys.path.insert(0, str(source_root))
    import torch

    from streaming.vulkan_bridge import VulkanNativeBridge
    from viewer.cuda_vulkan_interop import CudaVulkanImageImporter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    bridge = VulkanNativeBridge.load()
    encoder = bridge.create_encoder(
        width=args.width,
        height=args.height,
        fps=args.fps,
        target_bitrate=38_000_000,
        peak_bitrate=45_000_000,
        hevc=args.hevc,
    )
    writer = CudaVulkanImageImporter()
    packets = 0
    try:
        y = torch.full((args.height, args.width, 1), 81, dtype=torch.uint8, device="cuda")
        uv = torch.empty((args.height // 2, args.width // 2, 2), dtype=torch.uint8, device="cuda")
        uv[..., 0].fill_(90)
        uv[..., 1].fill_(240)
        for timestamp in range(args.frames):
            frame = encoder.acquire_frame()
            print(
                "vulkan_ffmpeg_cuda_submit_smoke: frame "
                f"slot={frame.slot_id} layouts={frame.layout[0]},{frame.layout[1]} "
                f"offsets={frame.memory_offset[0]},{frame.memory_offset[1]} "
                f"timeline={frame.semaphore_value[0]},{frame.semaphore_value[1]}"
            )
            ready_value = writer.write_ffmpeg_nv12_frame(y, uv, frame)
            encoder.submit_frame(
                frame,
                timestamp=timestamp,
                ready_semaphore=int(frame.external_semaphore_handle[0]),
                ready_value=ready_value,
            )
            while encoder.read_packet() is not None:
                packets += 1
        writer.synchronize()
        encoder.flush()
        while encoder.read_packet() is not None:
            packets += 1
        if packets < 1:
            raise RuntimeError("Vulkan Video accepted frames but produced no encoded packet")
        print(
            "vulkan_ffmpeg_cuda_submit_smoke: PASS "
            f"codec={'hevc' if args.hevc else 'h264'} size={args.width}x{args.height} "
            f"frames={args.frames} packets={packets} gpu_to_cpu=False stdin_raw=False"
        )
    finally:
        writer.close()
        encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
