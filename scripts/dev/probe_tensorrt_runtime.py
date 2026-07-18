from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(ROOT / "models" / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_294x518.onnx"))
    parser.add_argument("--create-session", action="store_true")
    args = parser.parse_args()

    from stereo_runtime.depth_trt_provider import candidate_tensorrt_lib_dirs, ensure_tensorrt_dll_path

    print("[TensorRT DLL candidates]", flush=True)
    for path in candidate_tensorrt_lib_dirs():
        print(f"- {path} exists={path.exists()} nvinfer_10={(path / 'nvinfer_10.dll').exists()}", flush=True)

    added = ensure_tensorrt_dll_path()
    print(f"[TensorRT DLL PATH added] {added}", flush=True)

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    print(f"[ORT] version={ort.__version__}", flush=True)
    print(f"[ORT] available={ort.get_available_providers()}", flush=True)

    onnx_path = Path(args.onnx)
    if not args.create_session:
        print("[ORT] skip session; pass --create-session to test real TensorRT EP activation", flush=True)
        return

    if not onnx_path.exists():
        print(f"[ORT] skip session, ONNX not found: {onnx_path}", flush=True)
        return

    session = ort.InferenceSession(
        str(onnx_path),
        providers=[
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    print(f"[ORT] active={session.get_providers()}", flush=True)


if __name__ == "__main__":
    main()
