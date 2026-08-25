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
from dataclasses import asdict

from stereo_runtime.triton_runtime import probe_triton_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the shared NVIDIA/AMD Triton stereo backend")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0")
    args = parser.parse_args()
    info = probe_triton_runtime(args.device, force=True)
    print(json.dumps(asdict(info), ensure_ascii=False, indent=2))
    return 0 if info.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
