from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .probe import build_capability_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop2Stereo Python Vulkan runtime")
    parser.add_argument("--probe", action="store_true", help="Print a JSON capability report and exit")
    parser.add_argument("--version", action="store_true", help="Print the project version and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.version:
        print("desktop2steoro-vulkan 0.1.0")
        return 0
    if args.probe:
        print(json.dumps(build_capability_report(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print("Desktop2Stereo Vulkan migration scaffold is ready.")
    print("Run with --probe to inspect capabilities.")
    return 0
