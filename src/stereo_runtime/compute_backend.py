from __future__ import annotations

from enum import StrEnum


NVIDIA_VENDOR_ID = 0x10DE


class StereoComputeBackend(StrEnum):
    # Keep the legacy enum name as an alias for the vendor-neutral Triton path.
    TRITON = "cuda_triton"
    CUDA_TRITON = "cuda_triton"
    VULKAN = "vulkan"
    TORCH_FALLBACK = "torch_fallback"


class StereoComputeBackendUnavailable(RuntimeError):
    pass


def resolve_stereo_compute_backend(
    requested: str = "auto",
    *,
    vendor_id: int | None,
    cuda_available: bool,
    vulkan_available: bool,
    triton_available: bool | None = None,
    triton_vendor: str | None = None,
) -> StereoComputeBackend:
    """Resolve the stereo synthesis backend without changing depth inference."""

    mode = str(requested or "auto").strip().lower().replace("-", "_")
    # ``None`` preserves the old call contract for NVIDIA callers. Runtime
    # callers should pass the result of probe_triton_runtime explicitly.
    triton_ready = (
        int(vendor_id or 0) == NVIDIA_VENDOR_ID and cuda_available
        if triton_available is None
        else bool(triton_available) and str(triton_vendor or "").lower() in {"nvidia", "amd"}
    )
    if mode in {"auto", "vendor", "vendor_default"}:
        if triton_ready:
            return StereoComputeBackend.TRITON
        if vulkan_available:
            return StereoComputeBackend.VULKAN
        return StereoComputeBackend.TORCH_FALLBACK

    if mode in {"vulkan", "vulkan_compute"}:
        if not vulkan_available:
            raise StereoComputeBackendUnavailable(
                "Vulkan Compute was explicitly requested but is unavailable"
            )
        return StereoComputeBackend.VULKAN

    if mode in {"cuda", "cuda_triton", "triton"}:
        if not triton_ready:
            raise StereoComputeBackendUnavailable(
                "Triton stereo synthesis requires a successful NVIDIA or AMD Triton runtime probe"
            )
        return StereoComputeBackend.TRITON

    raise ValueError(f"unknown stereo compute backend: {requested!r}")
