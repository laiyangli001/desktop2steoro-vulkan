from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_float(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(reports: dict[str, dict], output_path: Path) -> None:
    lines = [
        "# Depth Backend Environment Benchmark",
        "",
        "This report compares the stable `python3` environment and the experimental `python-cu13` environment.",
        "",
        "## Environments",
        "",
        "| Env | Python | Torch | Torch CUDA | ONNX Runtime | GPU | ORT Providers |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, report in reports.items():
        env = report.get("environment", {})
        lines.append(
            "| {name} | `{python}` | `{torch}` | `{torch_cuda}` | `{ort}` | `{gpu}` | `{providers}` |".format(
                name=name,
                python=env.get("python"),
                torch=env.get("torch"),
                torch_cuda=env.get("torch_cuda"),
                ort=env.get("onnxruntime", env.get("onnxruntime_error")),
                gpu=env.get("gpu"),
                providers=", ".join(env.get("ort_available_providers", [])),
            )
        )

    lines.extend(
        [
            "",
            "## Backend Timings",
            "",
            "| Env | Backend | Status | Setup ms | Warmup mean ms | Inference mean ms | Preprocess mean ms | Model mean ms | Postprocess mean ms | Inference median ms | Inference p90 ms | Depth backend | Runtime | Execution provider | IOBinding | DLPack | Output device | TensorRT DLL dirs | Error |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, report in reports.items():
        for backend, result in report.get("backends", {}).items():
            timings = result.get("timings", {})
            profile = result.get("profile") or {}
            provider = result.get("provider") or {}
            warmup = result.get("warmup", {})
            lines.append(
                "| {env} | `{backend}` | {status} | {setup} | {warmup_mean} | {mean} | {preprocess} | {model} | {postprocess} | {median} | {p90} | `{depth_backend}` | `{runtime}` | `{ep}` | `{iob}` | `{dlpack}` | `{outdev}` | `{trt_dirs}` | {error} |".format(
                    env=name,
                    backend=backend,
                    status=result.get("status"),
                    setup=format_float(result.get("setup_ms")),
                    warmup_mean=format_float(warmup.get("mean_ms")),
                    mean=format_float(timings.get("mean_ms")),
                    preprocess=format_float((profile.get("preprocess") or {}).get("mean_ms")),
                    model=format_float((profile.get("model") or {}).get("mean_ms")),
                    postprocess=format_float((profile.get("postprocess") or {}).get("mean_ms")),
                    median=format_float(timings.get("median_ms")),
                    p90=format_float(timings.get("p90_ms")),
                    depth_backend=provider.get("depth_backend"),
                    runtime=provider.get("runtime"),
                    ep=provider.get("execution_provider"),
                    iob=provider.get("io_binding"),
                    dlpack=provider.get("dlpack"),
                    outdev=provider.get("output_device"),
                    trt_dirs=", ".join(provider.get("trt_lib_dirs") or []),
                    error=(result.get("error") or "").replace("|", "\\|"),
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `tensorrt` is expected to use `TensorrtExecutionProvider`; if it fails, the failure is a real blocker for the TensorRT path.",
            "- TensorRT DLLs are auto-discovered from the active Python env and from `python3/Lib/site-packages/tensorrt_libs`, then prepended to `PATH` before ORT session creation.",
            "- `setup_ms` includes provider/session creation and TensorRT engine build/load. `Inference mean ms` reuses the same provider/session.",
            "- `preprocess`, `model`, and `postprocess` are measured inside provider prediction. For ORT backends, CPU numpy conversion is currently included in preprocess, and output retrieval is included in model.",
            "- `onnx_cuda_iobinding` must use ONNX Runtime CUDA with IOBinding enabled.",
            "- `pytorch_cuda` is the correctness and fallback baseline.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--out-dir", default="outputs/env_depth_backend_compare")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backend", action="append", default=None)
    parser.add_argument("--python3", default=str(ROOT / "python3" / "python.exe"))
    parser.add_argument("--python-cu13", default=str(ROOT / "python-cu13" / "python.exe"))
    args = parser.parse_args()

    envs = {
        "python3": Path(args.python3),
        "python-cu13": Path(args.python_cu13),
    }
    out_root = Path(args.out_dir)
    reports: dict[str, dict] = {}
    for name, python_exe in envs.items():
        if not python_exe.exists():
            reports[name] = {"environment": {"python": str(python_exe), "error": "python executable not found"}, "backends": {}}
            continue
        env_out = out_root / name
        command = [
            str(python_exe),
            "-B",
            str(ROOT / "scripts" / "benchmark" / "bench_depth_backends.py"),
            "--rgb",
            str(args.rgb),
            "--out-dir",
            str(env_out),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
        ]
        for backend in args.backend or ["tensorrt", "onnx_cuda_iobinding", "pytorch_cuda"]:
            command.extend(["--backend", backend])
        print(f"[env] {name}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        report_path = env_out / "depth_backend_bench.json"
        if report_path.exists():
            reports[name] = load_json(report_path)
            reports[name]["exit_code"] = result.returncode
        else:
            reports[name] = {"environment": {"python": str(python_exe)}, "backends": {}, "exit_code": result.returncode}

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "env_depth_backend_compare.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(reports, out_root / "env_depth_backend_compare.md")
    print(f"[done] wrote {out_root / 'env_depth_backend_compare.md'}", flush=True)


if __name__ == "__main__":
    main()
