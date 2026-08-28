from __future__ import annotations

from enum import StrEnum


NVIDIA_VENDOR_ID = 0x10DE


class StereoComputeBackend(StrEnum):
    # Keep the legacy enum name as an alias for the vendor-neutral Triton path.
    TRITON = "cuda_triton"
    CUDA_TRITON = "cuda_triton"
    VULKAN = "vulkan"
    OPENGL = "opengl"
    TORCH_FALLBACK = "torch_fallback"


class StereoComputeBackendUnavailable(RuntimeError):
    pass


def probe_opengl_stereo_backend() -> tuple[bool, str]:
    """Probe the distinct stereo-compute backend, not the stream upload fallback."""
    try:
        from streaming.opengl_stream_backend import OpenGLFallbackBackend
    except Exception as exc:
        return False, f"OpenGL module unavailable: {type(exc).__name__}: {exc}"
    if OpenGLFallbackBackend is None:
        return False, "OpenGL stream fallback class is unavailable"
    return False, "OpenGL stream fallback is output-only; stereo compute backend is not implemented"


def resolve_stereo_compute_backend(
    requested: str = "auto",
    *,
    vendor_id: int | None,
    cuda_available: bool,
    vulkan_available: bool,
    triton_available: bool | None = None,
    triton_vendor: str | None = None,
    opengl_available: bool = False,
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
        # Prefer the vendor-native GPU kernel when its runtime probe succeeds.
        # The Vulkan layered backend currently performs a host-visible 4K
        # readback for local output, which can reduce NVIDIA synthesis to about
        # one frame per second. Intel and other non-Triton devices continue to
        # use the cross-vendor Vulkan path.
        if triton_ready:
            return StereoComputeBackend.TRITON
        if vulkan_available:
            return StereoComputeBackend.VULKAN
        if opengl_available:
            return StereoComputeBackend.OPENGL
        return StereoComputeBackend.TORCH_FALLBACK

    if mode in {"vulkan", "vulkan_compute"}:
        if not vulkan_available:
            raise StereoComputeBackendUnavailable(
                "Vulkan Compute was explicitly requested but is unavailable"
            )
        return StereoComputeBackend.VULKAN

    if mode in {"opengl", "opengl_compute", "gl"}:
        if not opengl_available:
            raise StereoComputeBackendUnavailable(
                "OpenGL stereo synthesis was requested but is unavailable"
            )
        return StereoComputeBackend.OPENGL

    if mode in {"cuda", "cuda_triton", "triton"}:
        if not triton_ready:
            raise StereoComputeBackendUnavailable(
                "Triton stereo synthesis requires a successful NVIDIA or AMD Triton runtime probe"
            )
        return StereoComputeBackend.TRITON

    raise ValueError(f"unknown stereo compute backend: {requested!r}")
