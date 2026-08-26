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
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
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
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


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
]
