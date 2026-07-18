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
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--session-timeout", type=float, default=30.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be greater than zero")

    presenter = OpenXrVulkanPresenter(
        OpenXrVulkanConfig(render_scale=args.render_scale)
    )
    try:
        presenter.initialize()
        device = presenter.vulkan.device_info
        dimensions = [f"{eye.width}x{eye.height}" for eye in presenter.swapchains]
        print(
            f"OpenXR Vulkan initialized: GPU={device.name}, "
            f"Vulkan={device.api_version_text}, eyes={dimensions}"
        )

        deadline = time.monotonic() + args.session_timeout
        while presenter.frame_count < args.frames:
            if not presenter.run_frame():
                break
            if not presenter.session_running and time.monotonic() >= deadline:
                print("OpenXR session did not enter READY before timeout.", file=sys.stderr)
                return 3
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
