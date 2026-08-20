from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xr_viewer.core_openxr_vulkan import OpenXrVulkanConfig, OpenXrVulkanPresenter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a clear-color stereo projection layer through OpenXR Vulkan."
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Run for this many seconds after the OpenXR session becomes READY.",
    )
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--filament-scene-exposure", type=float, default=0.0)
    parser.add_argument("--filament-skybox-brightness", type=float, default=1.0)
    parser.add_argument("--session-timeout", type=float, default=30.0)
    parser.add_argument(
        "--filament-bridge",
        type=Path,
        default=None,
        help="Enable Filament rendering with a platform bridge library.",
    )
    parser.add_argument(
        "--filament-glb",
        type=Path,
        default=None,
        help="GLB file to load when Filament rendering is enabled.",
    )
    parser.add_argument(
        "--filament-profile",
        "--profile",
        type=Path,
        default=None,
        help="Environment profile containing the active view pose.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be greater than zero")
    if args.seconds is not None and args.seconds <= 0:
        raise SystemExit("--seconds must be greater than zero")
    if args.filament_glb and not args.filament_bridge:
        raise SystemExit("--filament-glb requires --filament-bridge")
    for option_name, path in (
        ("--filament-bridge", args.filament_bridge),
        ("--filament-glb", args.filament_glb),
        ("--filament-profile", args.filament_profile),
    ):
        if path is not None and not path.is_file():
            raise SystemExit(f"{option_name} does not exist: {path}")

    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(
            render_scale=args.render_scale,
            filament_bridge_path=str(args.filament_bridge)
            if args.filament_bridge
            else None,
            filament_glb_path=str(args.filament_glb) if args.filament_glb else None,
            filament_profile_path=(
                str(args.filament_profile) if args.filament_profile else None
            ),
            filament_scene_exposure_ev=args.filament_scene_exposure,
            filament_skybox_brightness=args.filament_skybox_brightness,
        )
    )
    try:
        presenter.initialize()
        device = presenter.vulkan.device_info
        dimensions = [f"{eye.width}x{eye.height}" for eye in presenter.swapchains]
        print(
            f"OpenXR Vulkan initialized: GPU={device.name}, "
            f"Vulkan={device.api_version_text}, eyes={dimensions}"
        )

        ready_deadline = time.monotonic() + args.session_timeout
        run_deadline = None
        while args.seconds is not None or presenter.frame_count < args.frames:
            if not presenter.run_frame():
                break
            now = time.monotonic()
            if presenter.session_running and run_deadline is None and args.seconds is not None:
                run_deadline = now + args.seconds
            if not presenter.session_running and now >= ready_deadline:
                print("OpenXR session did not enter READY before timeout.", file=sys.stderr)
                return 3
            if run_deadline is not None and now >= run_deadline:
                break
        print(f"Submitted {presenter.frame_count} OpenXR Vulkan frames.")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if type(exc).__name__ == "FormFactorUnavailableError":
            print(
                "Connect and wake the headset, then verify the active OpenXR runtime.",
                file=sys.stderr,
            )
            return 2
        return 1
    finally:
        presenter.close()


if __name__ == "__main__":
    raise SystemExit(main())
