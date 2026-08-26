from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from stereo_runtime.depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
)
from stereo_runtime.depth_upsample import DepthUpsampleMode


def is_directml_available() -> bool:
    """Return whether torch-directml exposes at least one usable device."""
    try:
        import torch_directml

        return bool(
            torch_directml.is_available()
            and int(torch_directml.device_count()) > 0
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False


def directml_device(index: int = 0) -> Any:
    import torch_directml

    if not is_directml_available():
        raise RuntimeError("torch-directml is unavailable")
    count = int(torch_directml.device_count())
    if index < 0 or index >= count:
        raise IndexError(f"DirectML device index {index} is out of range")
    return torch_directml.device(index)


def directml_device_name(index: int = 0) -> str | None:
    try:
        import torch_directml

        if not is_directml_available():
            return None
        return str(torch_directml.device_name(index)).strip().rstrip("\x00")
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _probe_representative_operators(device: Any) -> dict[str, dict[str, Any]]:
    """Probe representative model operators without claiming full model coverage."""
    import torch.nn.functional as F

    results: dict[str, dict[str, Any]] = {}
    x = torch.ones((1, 3, 8, 8), device=device, dtype=torch.float32)

    def attempt(name: str, operation) -> None:
        try:
            output = operation()
            observed = str(getattr(output, "device", ""))
            directml_resident = observed.lower().startswith("privateuseone")
            results[name] = {
                "supported": True,
                "device": observed,
                "implicit_cpu_fallback": not directml_resident,
            }
        except Exception as exc:
            results[name] = {
                "supported": False,
                "device": None,
                "implicit_cpu_fallback": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    attempt("conv2d", lambda: F.conv2d(x, torch.ones((4, 3, 3, 3), device=device), padding=1))
    attempt("interpolate", lambda: F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False))
    attempt("layer_norm", lambda: F.layer_norm(x, x.shape[-2:]))
    attempt("matmul", lambda: torch.matmul(torch.ones((8, 8), device=device), torch.ones((8, 8), device=device)))
    attempt("softmax", lambda: torch.softmax(x, dim=1))
    return results


def probe_directml_capabilities(index: int = 0) -> dict[str, Any]:
    """Probe DirectML runtime, representative operators, and conservative interop status.

    The operator probe is representative only; it is not a claim that the
    selected depth model is fully supported until that model is loaded.
    """
    report: dict[str, Any] = {
        "backend": "directml",
        "available": False,
        "device_count": 0,
        "device_name": None,
        "device": None,
        "device_create": False,
        "basic_ops": False,
        "implicit_cpu_fallback": None,
        "model_operator_support": "not_probed",
        "representative_operator_support": {},
        "d3d11_shared_resource": "not_implemented",
        "adapter_match": "not_probed",
        "adapter_identity": {
            "producer_luid": None,
            "consumer_luid": None,
            "status": "requires_matching_adapter_luid",
        },
        "shared_resource_capability": "not_probed",
        "vulkan_external_memory": "not_probed",
        "gpu_copy_count": None,
        "zero_copy": False,
    }
    try:
        import torch_directml

        count = int(torch_directml.device_count())
        report["device_count"] = count
        if not bool(torch_directml.is_available()) or count <= index:
            report["reason"] = "torch-directml reports no usable device"
            return report
        device = torch_directml.device(index)
        report["device"] = str(device)
        report["device_name"] = directml_device_name(index)
        report["device_create"] = True
        try:
            value = torch.ones((1,), device=device, dtype=torch.float32)
            result = value + 1.0
            observed = str(getattr(result, "device", ""))
            report["basic_ops"] = True
            report["implicit_cpu_fallback"] = not observed.lower().startswith("privateuseone")
            report["observed_device"] = observed
            report["representative_operator_support"] = _probe_representative_operators(device)
            report["model_operator_support"] = "representative_only"
            if report["implicit_cpu_fallback"]:
                report["reason"] = "basic DirectML operation returned a non-DirectML device"
            else:
                report["available"] = True
        except Exception as exc:
            report["reason"] = f"basic DirectML operation failed: {type(exc).__name__}: {exc}"
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    return report


class _DirectMLInfoMixin:
    def _mark_directml_info(self, info: DepthProviderInfo) -> DepthProviderInfo:
        return replace(
            info,
            depth_backend="directml",
            runtime="transformers-directml",
            execution_provider="DirectML PyTorch",
            fallback_reason=(
                None
                if is_directml_available()
                else "torch-directml is unavailable"
            ),
            output_device=str(self.device),
        )

    def predict_profile(self, rgb: torch.Tensor):
        """Reject implicit CPU model output so AutoDepthProvider can fall back."""
        try:
            result = super().predict_profile(rgb)
        except Exception as exc:
            self.info = replace(
                self.info,
                fallback_reason=(
                    f"DirectML inference failed: {type(exc).__name__}: {exc}"
                ),
            )
            raise
        depth = getattr(result, "depth", None)
        observed = str(getattr(depth, "device", "") or "")
        if not observed.lower().startswith("privateuseone"):
            reason = (
                "DirectML inference returned a non-DirectML tensor "
                f"({observed or 'unknown device'}); implicit CPU fallback rejected"
            )
            self.info = replace(self.info, fallback_reason=reason)
            raise RuntimeError(reason)
        return result


class DistillAnyDepthBaseDirectML(_DirectMLInfoMixin, DistillAnyDepthBase518):
    def __init__(
        self,
        *,
        device: Any | None = None,
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        resolved_device = directml_device() if device is None else device
        super().__init__(
            device=resolved_device,
            cache_dir=cache_dir,
            dtype=dtype,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        self.info = self._mark_directml_info(self.info)


class GenericAutoDepthDirectMLProvider(
    _DirectMLInfoMixin,
    GenericAutoDepthProvider,
):
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: Any | None = None,
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = 518,
        patch_size: int | None = 14,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        resolved_device = directml_device() if device is None else device
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            device=resolved_device,
            cache_dir=cache_dir,
            dtype=dtype,
            depth_resolution=depth_resolution,
            patch_size=patch_size,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        self.info = self._mark_directml_info(self.info)


def create_directml_provider(
    *,
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    model_name: str | None = None,
    device: Any | None = None,
    cache_dir: str | Path | None = None,
    depth_resolution: int = 518,
    patch_size: int | None = 14,
    local_files_only: bool = True,
    force_download: bool = False,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
):
    resolved_device = directml_device() if device is None else device
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
        return GenericAutoDepthDirectMLProvider(
            model_id=model_id,
            model_name=model_name,
            device=resolved_device,
            cache_dir=cache_dir,
            depth_resolution=depth_resolution,
            patch_size=patch_size,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
    return DistillAnyDepthBaseDirectML(
        device=resolved_device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
        depth_upsample=depth_upsample,
        depth_upsample_edge_strength=depth_upsample_edge_strength,
    )


__all__ = [
    "DistillAnyDepthBaseDirectML",
    "GenericAutoDepthDirectMLProvider",
    "create_directml_provider",
    "directml_device",
    "directml_device_name",
    "is_directml_available",
    "probe_directml_capabilities",
]
