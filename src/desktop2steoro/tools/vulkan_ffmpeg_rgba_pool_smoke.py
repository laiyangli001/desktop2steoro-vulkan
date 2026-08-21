"""Validate the native FFmpeg-owned single-plane RGBA producer pool."""

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
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(args.ffmpeg_bin).resolve()))
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    from streaming.vulkan_bridge import VulkanNativeBridge

    bridge = VulkanNativeBridge.load()
    encoder = bridge.create_encoder(
        width=args.width, height=args.height, fps=args.fps,
        target_bitrate=38_000_000, peak_bitrate=45_000_000,
    )
    frame = None
    try:
        frame = encoder.acquire_rgba_frame()
        if frame.plane_count != 1 or frame.format[0] not in (37, 43):
            raise RuntimeError(f"unexpected RGBA Vulkan frame: planes={frame.plane_count} format={frame.format[0]}")
        print(
            "vulkan_ffmpeg_rgba_pool_smoke: PASS "
            f"size={frame.width}x{frame.height} format={frame.format[0]} "
            f"layout={frame.layout[0]} timeline={frame.semaphore_value[0]}"
        )
    finally:
        if frame is not None:
            encoder._close_exported_handles(frame)
            encoder.release_rgba_frame()
        encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
