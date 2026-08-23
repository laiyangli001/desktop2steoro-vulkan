"""Inspect the published Intel Windows native runtime bundle.

This is a diagnostics entry point, not a build step.  It validates the
feature-specific DLL locations and their published hashes, then asks each
optional Python adapter for its runtime capability report.  Use ``--strict``
on an Intel Windows target to make missing native/runtime capabilities fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from desktop2stereo.stereo_runtime.providers.intel.d3d11_sbs_surface import (  # noqa: E402
    probe_d3d11_sbs_surface,
)
from desktop2stereo.stereo_runtime.providers.intel.native_artifacts import (  # noqa: E402
    native_artifact_directories,
)
from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import (  # noqa: E402
    probe_onevpl_d3d11,
)
from desktop2stereo.stereo_runtime.providers.intel.openvino_remote import (  # noqa: E402
    probe_openvino_remote_tensor,
)


def _repository_root() -> Path:
    return _SRC_ROOT.parent


def _published_root() -> Path:
    root = _repository_root()
    return root / "src" / "desktop2stereo" / "stereo_runtime" / "providers" / "intel" / "native"


def _published_files() -> dict[str, Path]:
    root = _repository_root() / "src" / "desktop2stereo"
    return {
        "capture": root / "capture" / "native" / "desktop_duplication" / "d2s_desktop_duplication.dll",
        "final_sbs_surface": root / "stereo_runtime" / "providers" / "intel" / "native" / "d3d11_sbs_surface" / "d2s_d3d11_sbs_surface.dll",
        "encoder": root / "stereo_runtime" / "providers" / "intel" / "native" / "onevpl_d3d11_encoder" / "d2s_onevpl_d3d11_encoder.dll",
        "onevpl_runtime": root / "stereo_runtime" / "providers" / "intel" / "native" / "onevpl_d3d11_encoder" / "libvpl.dll",
        "inference": root / "stereo_runtime" / "providers" / "intel" / "native" / "openvino_d3d11_bridge" / "d2s_openvino_d3d11_bridge.dll",
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _artifact_report() -> dict[str, Any]:
    manifest_path = _published_root() / "manifest.json"
    hashes_path = _published_root() / "sha256.json"
    expected: dict[str, str] = {}
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashes_path.is_file():
        for entry in json.loads(hashes_path.read_text(encoding="utf-8")):
            expected[str(entry["file"])] = str(entry["sha256"]).upper()

    files = {}
    for key, path in _published_files().items():
        actual = _hash_file(path) if path.is_file() else None
        files[key] = {
            "path": str(path),
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else 0,
            "sha256": actual,
            "expected_sha256": expected.get(path.name),
            "hash_match": actual is not None and actual == expected.get(path.name),
        }
    return {
        "published_root": str(_published_root()),
        "search_directories": [str(path) for path in native_artifact_directories()],
        "manifest": manifest,
        "files": files,
        "all_files_present": all(item["exists"] for item in files.values()),
        "all_hashes_match": all(item["hash_match"] for item in files.values()),
    }


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": sys.platform,
        "os_name": os.name,
        "artifacts": _artifact_report(),
        "desktop_duplication": {"available": False, "reason": "Windows DXGI probe not run"},
        "openvino_remote_tensor": None,
        "onevpl_d3d11": None,
        "d3d11_sbs_surface": None,
    }
    if os.name == "nt":
        from desktop2stereo.capture.backends.windows_desktop_duplication import probe

        report["desktop_duplication"] = probe()
        report["openvino_remote_tensor"] = probe_openvino_remote_tensor().__dict__
        report["onevpl_d3d11"] = probe_onevpl_d3d11()
        report["d3d11_sbs_surface"] = probe_d3d11_sbs_surface()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail when published files or Windows capabilities are unavailable")
    args = parser.parse_args(argv)
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not args.strict:
        return 0
    if not report["artifacts"]["all_files_present"] or not report["artifacts"]["all_hashes_match"]:
        return 1
    if os.name != "nt":
        return 1
    capture = report["desktop_duplication"]
    inference = report["openvino_remote_tensor"]
    encoder = report["onevpl_d3d11"]
    surface = report["d3d11_sbs_surface"]
    if not capture.get("available") or not inference or not inference.get("zero_copy_ready"):
        return 1
    if not encoder.get("available") or not surface.get("available"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
