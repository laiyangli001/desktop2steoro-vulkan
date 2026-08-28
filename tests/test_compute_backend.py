from __future__ import annotations

import pytest

from stereo_runtime.compute_backend import (
    NVIDIA_VENDOR_ID,
    StereoComputeBackend,
    StereoComputeBackendUnavailable,
    probe_opengl_stereo_backend,
    resolve_stereo_compute_backend,
)


def test_auto_prefers_nvidia_triton_before_vulkan():
    assert resolve_stereo_compute_backend(
        vendor_id=NVIDIA_VENDOR_ID,
        cuda_available=True,
        vulkan_available=True,
    ) is StereoComputeBackend.TRITON


def test_auto_uses_vulkan_for_amd_and_intel():
    for vendor_id in (0x1002, 0x8086):
        assert resolve_stereo_compute_backend(
            vendor_id=vendor_id,
            cuda_available=False,
            vulkan_available=True,
        ) is StereoComputeBackend.VULKAN


def test_auto_prefers_amd_triton_when_probe_succeeds():
    assert resolve_stereo_compute_backend(
        vendor_id=0x1002,
        cuda_available=True,
        vulkan_available=True,
        triton_available=True,
        triton_vendor="amd",
    ) is StereoComputeBackend.TRITON


def test_auto_uses_vulkan_when_amd_triton_probe_fails():
    assert resolve_stereo_compute_backend(
        vendor_id=0x1002,
        cuda_available=True,
        vulkan_available=True,
        triton_available=False,
        triton_vendor="amd",
    ) is StereoComputeBackend.VULKAN


def test_auto_does_not_route_unknown_vendor_to_triton():
    assert resolve_stereo_compute_backend(
        vendor_id=0x8086,
        cuda_available=True,
        vulkan_available=True,
        triton_available=True,
        triton_vendor="intel",
    ) is StereoComputeBackend.VULKAN


def test_explicit_triton_rejects_unknown_vendor():
    with pytest.raises(StereoComputeBackendUnavailable):
        resolve_stereo_compute_backend(
            "triton",
            vendor_id=0x8086,
            cuda_available=True,
            vulkan_available=True,
            triton_available=True,
            triton_vendor="intel",
        )


def test_explicit_vulkan_is_available_on_nvidia():
    assert resolve_stereo_compute_backend(
        "vulkan",
        vendor_id=NVIDIA_VENDOR_ID,
        cuda_available=True,
        vulkan_available=True,
    ) is StereoComputeBackend.VULKAN


def test_auto_can_select_opengl_only_when_probe_is_true():
    assert resolve_stereo_compute_backend(
        vendor_id=0x8086,
        cuda_available=False,
        vulkan_available=False,
        opengl_available=True,
    ) is StereoComputeBackend.OPENGL


def test_explicit_opengl_rejects_output_only_fallback():
    available, reason = probe_opengl_stereo_backend()
    assert available is False
    assert "output-only" in reason
    with pytest.raises(StereoComputeBackendUnavailable):
        resolve_stereo_compute_backend(
            "opengl",
            vendor_id=0x8086,
            cuda_available=False,
            vulkan_available=False,
            opengl_available=False,
        )


def test_explicit_vendor_backend_rejects_unavailable_cuda():
    with pytest.raises(StereoComputeBackendUnavailable):
        resolve_stereo_compute_backend(
            "cuda_triton",
            vendor_id=0x1002,
            cuda_available=False,
            vulkan_available=True,
            triton_available=False,
        )
