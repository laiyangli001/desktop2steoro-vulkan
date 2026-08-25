"""Run an Intel Windows Desktop Duplication -> OpenVINO -> oneVPL smoke test.

This is a target-machine validation entry point, not a build step. It keeps
one acquired Desktop Duplication texture alive while it is consumed by the
OpenVINO D3D11 provider and by the D3D11 VideoProcessor/oneVPL path, then
checks that all reported Adapter LUIDs agree. ``--frames`` can be increased
for a long-run test (for example, 1800 for roughly 30 minutes at 60 FPS).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_APP_ROOT = _SRC_ROOT / "desktop2stereo"
for _import_root in (_SRC_ROOT, _APP_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def run_smoke(*, model: str | Path, frames: int, output_index: int, fps: int) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("Intel native hardware smoke test requires Windows")

    from desktop2stereo.capture.backends.desktop_duplication_native import NativeDesktopDuplication
    from desktop2stereo.stereo_runtime.providers.intel.d3d11_sbs_surface import D3D11SbsSurface
    from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import OneVPLD3D11SurfaceEncoder
    from desktop2stereo.stereo_runtime.providers.intel.openvino_native_depth import OpenVINOD3D11DepthProvider

    capture = NativeDesktopDuplication(output_index=int(output_index))
    provider = surface = encoder = None
    started = time.perf_counter()
    acquired = encoded = 0
    adapter_luids: set[int] = set()
    dimensions: set[tuple[int, int]] = set()
    try:
        provider = OpenVINOD3D11DepthProvider(model_path=str(model), d3d11_device=int(capture.device or 0))
        for index in range(int(frames)):
            frame = capture.acquire_frame()
            if frame is None:
                continue
            try:
                acquired += 1
                width, height = int(frame.width), int(frame.height)
                dimensions.add((width, height))
                capture_luid = int(frame.adapter_luid)
                adapter_luids.add(capture_luid)
                if not capture_luid:
                    raise RuntimeError("Desktop Duplication returned an empty Adapter LUID")

                depth_result = provider.predict_native(frame)
                debug = dict(getattr(depth_result, "cuda_timing_events", {}) or {})
                provider_luid = int(debug.get("provider_adapter_luid", 0) or 0)
                if provider_luid and provider_luid != capture_luid:
                    raise RuntimeError(
                        "Desktop Duplication/OpenVINO Adapter LUID mismatch: "
                        f"capture=0x{capture_luid:016x} provider=0x{provider_luid:016x}"
                    )

                if surface is None:
                    surface = D3D11SbsSurface(width=width, height=height, d3d11_device=int(capture.device or 0))
                    if int(surface.adapter_luid) != capture_luid:
                        raise RuntimeError(
                            "Desktop Duplication/D3D11 surface Adapter LUID mismatch: "
                            f"capture=0x{capture_luid:016x} surface=0x{int(surface.adapter_luid):016x}"
                        )
                    encoder = OneVPLD3D11SurfaceEncoder(
                        width=width, height=height, fps=int(fps), bitrate=10_000_000,
                        d3d11_device=int(surface.device),
                    )
                    if int(encoder.adapter_luid) != capture_luid:
                        raise RuntimeError(
                            "Desktop Duplication/oneVPL Adapter LUID mismatch: "
                            f"capture=0x{capture_luid:016x} encoder=0x{int(encoder.adapter_luid):016x}"
                        )
                elif (surface.width, surface.height) != (width, height):
                    raise RuntimeError("Desktop Duplication frame dimensions changed during smoke test")

                surface.set_bgra_texture(int(frame.texture), adapter_luid=capture_luid)
                nv12_texture, nv12_width, nv12_height = surface.nv12_texture()
                if (nv12_width, nv12_height) != (width, height) or not nv12_texture:
                    raise RuntimeError("D3D11 VideoProcessor did not produce a valid NV12 surface")
                encoder.submit_nv12(nv12_texture, index)
                if encoder.read_packet():
                    encoded += 1
            finally:
                frame.release()
    finally:
        if encoder is not None:
            encoder.close()
        if surface is not None:
            surface.close()
        if provider is not None:
            provider.close()
        capture.close()

    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "frames_requested": int(frames),
        "frames_acquired": acquired,
        "packets_observed": encoded,
        "elapsed_seconds": round(elapsed, 3),
        "observed_fps": round(acquired / elapsed, 2),
        "dimensions": [list(item) for item in sorted(dimensions)],
        "adapter_luids": [f"0x{item:016x}" for item in sorted(adapter_luids)],
        "gpu_to_cpu": False,
        "zero_copy": False,
        "zero_copy_note": "depth output ABI is CPU-backed; this smoke test validates GPU input and same-adapter encoding",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="OpenVINO IR/XML model path")
    parser.add_argument("--frames", type=_positive_int, default=1)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--fps", type=_positive_int, default=60)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(model=args.model, frames=args.frames, output_index=args.output_index, fps=args.fps)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    report["ok"] = report["frames_acquired"] == report["frames_requested"] and report["packets_observed"] > 0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
