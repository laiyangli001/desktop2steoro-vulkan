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


def is_rocm_torch_available() -> bool:
    return getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()


def rocm_device_name(index: int = 0) -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return None


class _RocmInfoMixin:
    def _mark_rocm_info(self, info: DepthProviderInfo) -> DepthProviderInfo:
        return replace(
            info,
            depth_backend="pytorch_rocm",
            runtime="transformers-rocm",
            execution_provider="ROCm PyTorch",
            fallback_reason=None if is_rocm_torch_available() else "torch.version.hip is not available",
            output_device=str(self.device),
        )


class DistillAnyDepthBaseRocm(_RocmInfoMixin, DistillAnyDepthBase518):
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
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
        self.info = self._mark_rocm_info(self.info)


class GenericAutoDepthRocmProvider(_RocmInfoMixin, GenericAutoDepthProvider):
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "cuda",
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
        self.info = self._mark_rocm_info(self.info)


TorchRocmDepthProvider = DistillAnyDepthBaseRocm
GenericTorchRocmDepthProvider = GenericAutoDepthRocmProvider


def create_pytorch_rocm_provider(
    *,
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    model_name: str | None = None,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    depth_resolution: int = 518,
    patch_size: int | None = 14,
    local_files_only: bool = True,
    force_download: bool = False,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
):
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
        return GenericAutoDepthRocmProvider(
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
    return DistillAnyDepthBaseRocm(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
        depth_upsample=depth_upsample,
        depth_upsample_edge_strength=depth_upsample_edge_strength,
    )


__all__ = [
    "TorchRocmDepthProvider",
    "GenericTorchRocmDepthProvider",
    "DistillAnyDepthBaseRocm",
    "GenericAutoDepthRocmProvider",
    "create_pytorch_rocm_provider",
    "is_rocm_torch_available",
    "rocm_device_name",
]
