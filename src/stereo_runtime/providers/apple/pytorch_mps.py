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


def is_mps_torch_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


class _MpsInfoMixin:
    def _mark_mps_info(self, info: DepthProviderInfo) -> DepthProviderInfo:
        return replace(
            info,
            depth_backend="pytorch_mps",
            runtime="transformers-mps",
            execution_provider="Apple MPS PyTorch",
            fallback_reason=None if is_mps_torch_available() else "torch.backends.mps is not available",
            output_device=str(self.device),
        )


class DistillAnyDepthBaseMps(_MpsInfoMixin, DistillAnyDepthBase518):
    def __init__(
        self,
        *,
        device: str | torch.device = "mps",
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
        self.info = self._mark_mps_info(self.info)


class GenericAutoDepthMpsProvider(_MpsInfoMixin, GenericAutoDepthProvider):
    def __init__(
        self,
        *,
        model_id: str,
        model_name: str | None = None,
        device: str | torch.device = "mps",
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
        self.info = self._mark_mps_info(self.info)


TorchMpsDepthProvider = DistillAnyDepthBaseMps
GenericTorchMpsDepthProvider = GenericAutoDepthMpsProvider


def create_pytorch_mps_provider(
    *,
    model_id: str = DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    model_name: str | None = None,
    device: str | torch.device = "mps",
    cache_dir: str | Path | None = None,
    depth_resolution: int = 518,
    patch_size: int | None = 14,
    local_files_only: bool = True,
    force_download: bool = False,
    depth_upsample: DepthUpsampleMode = "bilinear",
    depth_upsample_edge_strength: float = 0.35,
):
    if model_id != DISTILL_ANY_DEPTH_BASE_MODEL_ID:
        return GenericAutoDepthMpsProvider(
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
    return DistillAnyDepthBaseMps(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
        depth_upsample=depth_upsample,
        depth_upsample_edge_strength=depth_upsample_edge_strength,
    )


__all__ = [
    "TorchMpsDepthProvider",
    "GenericTorchMpsDepthProvider",
    "DistillAnyDepthBaseMps",
    "GenericAutoDepthMpsProvider",
    "create_pytorch_mps_provider",
    "is_mps_torch_available",
]
