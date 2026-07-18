from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import warnings

import torch

from utils.cpu_warnings import describe_tensor, warn_cpu_operation

OnnxDtypeMode = Literal["auto", "fp16", "fp32"]

from .model_capabilities import FORCE_FP32_KEYWORDS
from .progress import activity_progress, status_write


@dataclass(frozen=True)
class ExportDtypeChoice:
    dtype: object
    name: str
    reason: str


@dataclass(frozen=True)
class OnnxExportResult:
    output_path: Path
    dtype_name: str
    dtype_reason: str
    probe_reason: str
    size_mb: float


def choose_export_dtype(model_id: str, device, requested: OnnxDtypeMode) -> tuple[object, str, str]:
    import torch

    if requested == "fp16":
        return torch.float16, "fp16", "requested fp16"
    if requested == "fp32":
        return torch.float32, "fp32", "requested fp32"
    model_lower = model_id.lower()
    if device.type != "cuda":
        return torch.float32, "fp32", "auto: non-CUDA device"
    if "infinidepth" in model_lower:
        return torch.float32, "fp32", "auto: InfiniDepth requires fp32"
    if any(keyword in model_lower for keyword in FORCE_FP32_KEYWORDS):
        return torch.float32, "fp32", "auto: model requires fp32"
    return torch.float16, "fp16", "auto: CUDA default"


def _extract_depth_output(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, dict) and "predicted_depth" in output:
        return output["predicted_depth"]
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(f"unsupported model output type: {type(output).__name__}")


def probe_model_dtype(model, *, device, dtype, height: int, width: int) -> tuple[bool, str]:
    import torch

    dummy = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    try:
        model.eval()
        with torch.no_grad():
            output = model(pixel_values=dummy)
        depth = _extract_depth_output(output).detach().float()
        if depth.numel() == 0:
            return False, "empty output"
        warn_cpu_operation(
            "stereo_runtime.probe_model_dtype",
            "depth validity/range .item() sync",
            detail=describe_tensor(depth),
            key=f"stereo_runtime_probe_model_dtype_cpu_stats_{dtype}",
        )
        if not torch.isfinite(depth).all().item():
            return False, "output contains NaN or Inf"
        abs_max = float(depth.abs().max().item())
        value_range = float((depth.max() - depth.min()).abs().item())
        if abs_max == 0.0:
            return False, "output is all zero"
        if value_range < 1e-7:
            return False, f"output dynamic range too small: {value_range:.3e}"
        return True, f"ok: shape={tuple(depth.shape)} abs_max={abs_max:.6g} range={value_range:.6g}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class DepthOnnxExportWrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        return _extract_depth_output(output)


class InfiniDepthOnnxExportWrapper(torch.nn.Module):
    def __init__(self, model, *, fp32: bool = False) -> None:
        super().__init__()
        self.model = model
        self.fp32 = bool(fp32)

    def forward(self, pixel_values):
        return self.model(pixel_values, fp32=self.fp32)


def _is_infinidepth_model(model_id: str) -> bool:
    return "infinidepth" in str(model_id).lower()


@contextmanager
def _quiet_onnx_export_warnings():
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"ONNX export mode is set to TrainingMode\.EVAL, but operator 'instance_norm' is set to train=True.*",
            category=UserWarning,
        )
        yield


def load_model_for_dtype(
    auto_model_cls,
    model_id: str,
    *,
    dtype,
    device,
    cache_dir: Path,
    local_files_only: bool,
    force_download: bool,
):
    import torch

    if _is_infinidepth_model(model_id):
        from stereo_runtime.model_impl.InfiniDepth.api import InfiniDepthModel

        from .depth_provider import _infinidepth_encoder_for_model, _resolve_hf_model_file

        model_path = _resolve_hf_model_file(
            model_id,
            cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        model = InfiniDepthModel(model_path=model_path, encoder=_infinidepth_encoder_for_model(model_id)).to(device, dtype=dtype)
        model.eval()
        return InfiniDepthOnnxExportWrapper(model, fp32=dtype != torch.float16).eval()

    model = auto_model_cls.from_pretrained(
        model_id,
        dtype=dtype,
        cache_dir=str(cache_dir),
        weights_only=True,
        local_files_only=local_files_only,
        force_download=force_download,
    ).to(device)
    if dtype == torch.float16:
        model.half()
    else:
        model.float()
    model.eval()
    return DepthOnnxExportWrapper(model).eval()


def export_depth_model_onnx(
    *,
    model_id: str,
    output_path: str | Path,
    cache_dir: str | Path,
    device: str = "cuda",
    height: int = 294,
    width: int = 518,
    dtype: OnnxDtypeMode = "auto",
    local_files_only: bool = False,
    force_download: bool = True,
    auto_model_cls=None,
) -> OnnxExportResult:
    import torch

    if auto_model_cls is None:
        from transformers import AutoModelForDepthEstimation

        auto_model_cls = AutoModelForDepthEstimation

    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype_obj, dtype_name, dtype_reason = choose_export_dtype(model_id, device_obj, dtype)
    output_path = Path(output_path)
    cache_dir = Path(cache_dir)

    with activity_progress(f"Loading model for ONNX export: {model_id}"):
        model = load_model_for_dtype(
            auto_model_cls,
            model_id,
            dtype=dtype_obj,
            device=device_obj,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
        )

    with activity_progress(f"Probing ONNX export dtype: {dtype_name}"):
        ok, probe_reason = probe_model_dtype(model, device=device_obj, dtype=dtype_obj, height=height, width=width)
    if dtype in ("auto", "fp16") and not ok and dtype_name == "fp16":
        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
        dtype_obj = torch.float32
        dtype_name = "fp32"
        dtype_reason = f"{dtype}: fp16 probe failed ({probe_reason}); fallback fp32"
        output_path = output_path.with_name(output_path.name.replace("model_fp16_", "model_fp32_"))
        with activity_progress(f"Reloading model for ONNX export: {dtype_name}"):
            model = load_model_for_dtype(
                auto_model_cls,
                model_id,
                dtype=dtype_obj,
                device=device_obj,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                force_download=False,
            )
        with activity_progress(f"Probing ONNX export dtype: {dtype_name}"):
            ok, probe_reason = probe_model_dtype(model, device=device_obj, dtype=dtype_obj, height=height, width=width)
    if not ok:
        raise RuntimeError(f"export dtype probe failed for {dtype_name}: {probe_reason}")

    dummy_input = torch.randn(1, 3, height, width, device=device_obj, dtype=dtype_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_write(f"正在导出 ONNX：{output_path.name}。这个步骤可能需要一段时间，进度请看上方进度条。")
    with activity_progress(f"Exporting ONNX: {output_path.name}"):
        with torch.no_grad(), _quiet_onnx_export_warnings():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=["pixel_values"],
                output_names=["predicted_depth"],
                do_constant_folding=True,
                export_params=True,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL,
                dynamo=False,
            )
    status_write(f"ONNX 导出完成：{output_path.name}。")

    return OnnxExportResult(
        output_path=output_path,
        dtype_name=dtype_name,
        dtype_reason=dtype_reason,
        probe_reason=probe_reason,
        size_mb=output_path.stat().st_size / (1024 * 1024),
    )
