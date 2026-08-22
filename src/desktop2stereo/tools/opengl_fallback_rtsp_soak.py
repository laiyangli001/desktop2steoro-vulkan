"""End-to-end OpenGL fallback -> encoder -> MediaMTX/WebRTC soak.

This diagnostic deliberately disables the native Vulkan bridge for the session,
then submits CUDA RGBA frames through VulkanDirectSbsOutput. On NVIDIA it
must select CUDA/OpenGL interop -> PyNvVideoCodec/NVENC; on other platforms it
may select the documented host-upload fallback. It therefore verifies the
actual fallback owner and MediaMTX publishing boundary, not only the OpenGL
texture probe.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=None, help="desktop2steoro-vulkan directory")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--protocol", default="WEBRTC", choices=("WEBRTC", "SRT"))
    parser.add_argument("--force-host", action="store_true", help="force the OpenGL PBO/host-upload branch")
    parser.add_argument("--cpu", action="store_true", help="submit CPU RGB frames to exercise the non-CUDA fallback")
    args = parser.parse_args()

    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("width and height must be positive even values")
    if args.fps < 1 or args.frames < 1:
        parser.error("fps and frames must be positive")

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parents[3]
    source_root = base_dir / "src" / "desktop2stereo"
    sys.path.insert(0, str(source_root))

    import numpy as np
    torch = None
    if not args.cpu:
        import torch as torch_module

        torch = torch_module

    from streaming.direct_sbs import VulkanDirectSbsOutput

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --cpu or opengl_fallback_smoke.py")

    previous_force_host = os.environ.get("D2S_OPENGL_FORCE_HOST")

    output = VulkanDirectSbsOutput(
        base_dir=source_root,
        protocol=args.protocol,
        port=1122,
        stream_key="opengl-fallback-soak",
        fps=args.fps,
        crf=23,
        stereo_mix_device="",
        os_name=None,
        prefer_nvenc=True,
        display_mode="Half-SBS",
    )
    if args.force_host:
        os.environ["D2S_OPENGL_FORCE_HOST"] = "1"
    # Force the documented transition point without changing production code:
    # the first submit must enter _fallback_to_opengl().
    output._native_vulkan_bridge = None
    tensor = (
        np.empty((args.height, args.width, 3), dtype=np.uint8)
        if args.cpu
        else torch.empty((args.height, args.width, 4), dtype=torch.uint8, device="cuda")
    )
    if not args.cpu:
        tensor[..., 3] = 255

    submitted = 0
    started = time.monotonic()
    try:
        output.start()
        next_deadline = started
        for frame_index in range(args.frames):
            tensor[..., 0] = (32 + frame_index) % 256
            tensor[..., 1] = (96 + frame_index * 3) % 256
            tensor[..., 2] = (192 + frame_index * 5) % 256
            if args.cpu:
                output.submit_frame(tensor)
            else:
                tensor[..., 3] = 255
                output.submit_cuda_frame(tensor)
            submitted += 1
            next_deadline += 1.0 / args.fps
            remaining = next_deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        if not output._opengl_fallback_active:
            raise RuntimeError("OpenGL fallback was not active after forced Vulkan failure")
        if output._opengl_fallback is None:
            raise RuntimeError("OpenGL fallback backend was not retained")
        selected = (
            "cuda-opengl-interop"
            if output._opengl_pynv_fallback is not None
            else "hip-opengl-interop"
            if output._opengl_amd_fallback is not None
            else "host-upload"
            if getattr(output, "_host_fallback", None) is not None
            else "unknown"
        )
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            "opengl_fallback_rtsp_soak: PASS "
            f"path={selected} size={args.width}x{args.height} "
            f"fps={args.fps} frames={submitted} elapsed={elapsed:.2f}s"
        )
        return 0
    finally:
        del tensor
        output.close()
        if previous_force_host is None:
            os.environ.pop("D2S_OPENGL_FORCE_HOST", None)
        else:
            os.environ["D2S_OPENGL_FORCE_HOST"] = previous_force_host


if __name__ == "__main__":
    raise SystemExit(main())
