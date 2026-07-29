"""Compare legacy and MIP screen sampling runtime captures pixel by pixel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stereo_runtime.screen_sampling_visual_regression import (
    compare_screen_sampling_capture_dirs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", required=True, type=Path)
    parser.add_argument("--mip-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage",
        choices=("03_vulkan_output", "06_openxr_projection"),
        default="06_openxr_projection",
    )
    parser.add_argument(
        "--no-verify-source",
        action="store_true",
        help="Do not compare the 03_vulkan_output input stage before the projection stage",
    )
    args = parser.parse_args()
    result = compare_screen_sampling_capture_dirs(
        args.legacy_dir,
        args.mip_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        verify_source=not args.no_verify_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "comparison output: "
        f"{(args.output_dir or args.mip_dir / 'screen_sampling_comparison').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
