from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem).strip("_") or "image"


def collect_rgb_files(args: argparse.Namespace) -> list[Path]:
    files = [Path(x) for x in args.rgb]
    if args.rgb_dir:
        root = Path(args.rgb_dir)
        if not root.exists():
            raise FileNotFoundError(f"rgb dir does not exist: {root}")
        if args.recursive:
            files.extend(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            files.extend(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", action="append", default=[], help="RGB image path; repeat for multiple files")
    parser.add_argument("--rgb-dir", default=None, help="directory containing RGB images")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--provider", action="append", choices=["distill_base_518", "distill_base_nvidia", "luma"], default=None)
    parser.add_argument("--depth-onnx", default=None)
    parser.add_argument("--no-pytorch-fallback", action="store_true")
    parser.add_argument("--require-tensorrt", action="store_true")
    parser.add_argument("--reference-depth", action="append", default=[], help="reference depth passed to each image")
    parser.add_argument("--out-dir", default="outputs/depth_batch")
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-cache-dir", default=None)
    parser.add_argument("--depth-local-only", action="store_true")
    parser.add_argument("--depth-force-download", action="store_true")
    args = parser.parse_args()

    rgb_files = collect_rgb_files(args)
    if not rgb_files:
        raise SystemExit("no RGB images found; pass --rgb or --rgb-dir")

    providers = args.provider or ["distill_base_518"]
    out_root = Path(args.out_dir)
    print(f"[info] images={len(rgb_files)} providers={','.join(providers)} out={out_root}", flush=True)

    failures = []
    for index, rgb_path in enumerate(rgb_files, start=1):
        image_out = out_root / safe_name(rgb_path)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "tools" / "generate_depth_map.py"),
            "--rgb",
            str(rgb_path),
            "--out-dir",
            str(image_out),
        ]
        for provider in providers:
            command.extend(["--provider", provider])
        for reference in args.reference_depth:
            command.extend(["--reference-depth", reference])
        if args.device:
            command.extend(["--device", args.device])
        if args.depth_cache_dir:
            command.extend(["--depth-cache-dir", args.depth_cache_dir])
        if args.depth_onnx:
            command.extend(["--depth-onnx", args.depth_onnx])
        if args.no_pytorch_fallback:
            command.append("--no-pytorch-fallback")
        if args.require_tensorrt:
            command.append("--require-tensorrt")
        if args.depth_local_only:
            command.append("--depth-local-only")
        if args.depth_force_download:
            command.append("--depth-force-download")

        print(f"[{index}/{len(rgb_files)}] {rgb_path} -> {image_out}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            failures.append({"rgb": str(rgb_path), "exit_code": result.returncode})

    if failures:
        print("[error] failed images:", flush=True)
        for failure in failures:
            print(f"  {failure['rgb']} exit={failure['exit_code']}", flush=True)
        raise SystemExit(1)

    print(f"[done] wrote batch depth reports to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
