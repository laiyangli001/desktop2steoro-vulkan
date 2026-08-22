from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from stereo_runtime.depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
)
from stereo_runtime.depth_upsample import DepthUpsampleMode


def is_xpu_torch_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except Exception:
        return False


def xpu_device_name(index: int = 0) -> str | None:
    if not is_xpu_torch_available():
        return None
    try:
        return torch.xpu.get_device_name(index)
    except Exception:
        return None


class _XpuInfoMixin:
    def _mark_xpu_info(self, info: DepthProviderInfo) -> DepthProviderInfo:
        return replace(
            info,
            depth_backend="pytorch_xpu",
            runtime="transformers-xpu",
            execution_provider="Intel XPU PyTorch",
            fallback_reason=None if is_xpu_torch_available() else "torch.xpu is not available",
            output_device=str(self.device),
        )


class DistillAnyDepthBaseXpu(_XpuInfoMixin, DistillAnyDepthBase518):
    def __init__(
        self,
        *,
        device: str | torch.device = "xpu",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        super().__init__(
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        self.info = self._mark_xpu_info(self.info)


class GenericAutoDepthXpuProvider(_XpuInfoMixin, GenericAutoDepthProvider):
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "xpu",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        depth_resolution: int = 518,
        patch_size: int | None = 14,
        local_files_only: bool = False,
        force_download: bool = False,
        depth_upsample: DepthUpsampleMode = "bilinear",
        depth_upsample_edge_strength: float = 0.35,
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            dtype=dtype,
            depth_resolution=depth_resolution,
            patch_size=patch_size,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
        self.info = self._mark_xpu_info(self.info)


TorchXpuDepthProvider = DistillAnyDepthBaseXpu
GenericTorchXpuDepthProvider = GenericAutoDepthXpuProvider


def create_pytorch_xpu_provider(
    *,
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    model_name: str | None = None,
    device: str | torch.device = "xpu",
    cache_dir: str | Path | None = None,
    depth_resolution: int = 518,
    patch_size: int | None = 14,
    local_files_only: bool = True,
    force_download: bool = False,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
):
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
        return GenericAutoDepthXpuProvider(
            model_id=model_id,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            depth_resolution=depth_resolution,
            patch_size=patch_size,
            local_files_only=local_files_only,
            force_download=force_download,
            depth_upsample=depth_upsample,
            depth_upsample_edge_strength=depth_upsample_edge_strength,
        )
    return DistillAnyDepthBaseXpu(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
        depth_upsample=depth_upsample,
        depth_upsample_edge_strength=depth_upsample_edge_strength,
    )


__all__ = [
    "TorchXpuDepthProvider",
    "GenericTorchXpuDepthProvider",
    "DistillAnyDepthBaseXpu",
    "GenericAutoDepthXpuProvider",
    "create_pytorch_xpu_provider",
    "is_xpu_torch_available",
    "xpu_device_name",
]
