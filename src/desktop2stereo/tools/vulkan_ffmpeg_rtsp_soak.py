"""End-to-end native Vulkan packet -> FFmpeg -> MediaMTX RTSP soak.

This diagnostic intentionally keeps the image path native and only sends
compressed H.264 packets through the muxer stdin. It does not represent the
production capture loop; it verifies the final local publishing boundary.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def _drain(process: subprocess.Popen, target: deque[str]) -> None:
    stream = process.stderr
    if stream is None:
        return
    while True:
        line = stream.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        target.append(line.rstrip())


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg-bin", required=True, help="FFmpeg bin directory")
    parser.add_argument("--mediamtx-bin", required=True)
    parser.add_argument("--mediamtx-config", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--url", default="rtsp://127.0.0.1:8554/live?pkt_size=1452")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    args = parser.parse_args()

    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("dimensions must be positive even values")
    if args.fps < 1 or args.duration_seconds <= 0:
        parser.error("fps and duration-seconds must be positive")

    ffmpeg_bin = Path(args.ffmpeg_bin).resolve()
    mediamtx_bin = Path(args.mediamtx_bin).resolve()
    mediamtx_config = Path(args.mediamtx_config).resolve()
    bridge_path = Path(args.bridge).resolve()
    if not all(path.is_file() for path in (mediamtx_bin, mediamtx_config, bridge_path)):
        raise FileNotFoundError("FFmpeg, MediaMTX, config, and bridge paths must exist")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(ffmpeg_bin))
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))

    import torch

    from streaming.vulkan_bridge import VulkanNativeBridge
    from viewer.cuda_vulkan_interop import CudaVulkanImageImporter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    mediamtx_log: deque[str] = deque(maxlen=80)
    mux_log: deque[str] = deque(maxlen=80)
    mediamtx = subprocess.Popen(
        [str(mediamtx_bin), str(mediamtx_config)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creationflags(),
    )
    mediamtx_thread = threading.Thread(
        target=_drain, args=(mediamtx, mediamtx_log), daemon=True
    )
    mediamtx_thread.start()
    mux: subprocess.Popen | None = None
    encoder = None
    importer = None
    sent_frames = 0
    sent_packets = 0
    try:
        time.sleep(0.25)
        if mediamtx.poll() is not None:
            raise RuntimeError(
                f"MediaMTX exited with {mediamtx.returncode}: "
                + "; ".join(mediamtx_log)
            )

        mux = subprocess.Popen(
            [
                str(ffmpeg_bin / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")),
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "nobuffer",
                "-f",
                "h264",
                "-r",
                str(args.fps),
                "-i",
                "pipe:0",
                "-c:v",
                "copy",
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-flush_packets",
                "1",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                "-pkt_size",
                "1452",
                args.url,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=False,
            creationflags=_creationflags(),
        )
        mux_thread = threading.Thread(target=_drain, args=(mux, mux_log), daemon=True)
        mux_thread.start()
        time.sleep(0.25)
        if mux.poll() is not None or mux.stdin is None:
            raise RuntimeError(
                f"FFmpeg muxer exited with {mux.returncode}: " + "; ".join(mux_log)
            )

        bridge = VulkanNativeBridge.load(bridge_path)
        encoder = bridge.create_encoder(
            width=args.width,
            height=args.height,
            fps=args.fps,
            target_bitrate=40_000_000,
            peak_bitrate=46_000_000,
        )
        importer = CudaVulkanImageImporter()
        tensor = torch.empty(
            (args.height, args.width, 4), dtype=torch.uint8, device="cuda"
        )
        tensor[..., 3] = 255
        frame_count = max(1, int(round(args.duration_seconds * args.fps)))
        deadline = time.monotonic()
        for timestamp in range(frame_count):
            frame = encoder.acquire_rgba_frame()
            tensor[..., 0] = (32 + timestamp) % 256
            tensor[..., 1] = (96 + timestamp * 3) % 256
            tensor[..., 2] = (192 + timestamp * 5) % 256
            ready_value = importer.write_ffmpeg_rgba_frame(tensor, frame)
            encoder.encode_rgba_frame(ready_value=ready_value, timestamp=timestamp)
            while True:
                packet = encoder.read_packet()
                if not packet:
                    break
                mux.stdin.write(packet)
                sent_packets += 1
            mux.stdin.flush()
            sent_frames += 1
            deadline += 1.0 / args.fps
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        encoder.flush()
        while True:
            packet = encoder.read_packet()
            if not packet:
                break
            mux.stdin.write(packet)
            sent_packets += 1
        mux.stdin.flush()
        if mux.poll() is not None:
            raise RuntimeError(
                f"FFmpeg muxer exited during publish: {mux.returncode}: "
                + "; ".join(mux_log)
            )
        if sent_frames != frame_count or sent_packets < 1:
            raise RuntimeError(
                f"publish incomplete: frames={sent_frames}/{frame_count} "
                f"packets={sent_packets}"
            )
        print(
            "vulkan_ffmpeg_rtsp_soak: PASS "
            f"size={args.width}x{args.height} fps={args.fps} "
            f"frames={sent_frames} packets={sent_packets} url={args.url}"
        )
        return 0
    finally:
        if importer is not None:
            importer.close()
        if encoder is not None:
            encoder.close()
        if mux is not None:
            if mux.stdin is not None:
                mux.stdin.close()
            try:
                mux.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mux.terminate()
                mux.wait(timeout=5)
        if mediamtx.poll() is None:
            mediamtx.terminate()
            try:
                mediamtx.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mediamtx.kill()
                mediamtx.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
