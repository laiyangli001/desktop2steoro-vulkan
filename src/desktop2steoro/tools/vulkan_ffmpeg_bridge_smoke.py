"""Validate native FFmpeg Vulkan device creation and NV12 frame-pool allocation.

This deliberately does not submit a frame.  Native frame submission remains
fail-closed until the external-memory and semaphore hand-off is implemented.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_dll_search_path(ffmpeg_bin: str | None) -> None:
    if not ffmpeg_bin:
        return
    directory = Path(ffmpeg_bin).resolve()
    if not directory.is_dir():
        raise ValueError(f"FFmpeg bin directory does not exist: {directory}")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(directory))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a native FFmpeg Vulkan encoder and acquire one NV12 frame"
    )
    parser.add_argument("--ffmpeg-bin", help="directory containing avcodec-63.dll")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hevc", action="store_true")
    parser.add_argument("--target-bitrate", type=int, default=38_000_000)
    parser.add_argument("--peak-bitrate", type=int, default=45_000_000)
    args = parser.parse_args()
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("NV12 dimensions must be positive even values")
    if args.fps < 1:
        parser.error("fps must be positive")

    _configure_dll_search_path(args.ffmpeg_bin)
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    from streaming.vulkan_bridge import VulkanNativeBridge

    bridge = VulkanNativeBridge.load()
    encoder = bridge.create_encoder(
        width=args.width,
        height=args.height,
        fps=args.fps,
        target_bitrate=args.target_bitrate,
        peak_bitrate=args.peak_bitrate,
        hevc=args.hevc,
    )
    try:
        frame = encoder.acquire_frame()
        if frame.plane_count != 2 or not frame.image[0] or not frame.image[1]:
            raise RuntimeError(
                f"expected two NV12 image planes, received {frame.plane_count}"
            )
        print(
            "vulkan_ffmpeg_bridge_smoke: PASS "
            f"codec={'hevc' if args.hevc else 'h264'} "
            f"size={frame.width}x{frame.height} planes={frame.plane_count} "
            f"formats={frame.format[0]},{frame.format[1]}"
        )
    finally:
        encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
