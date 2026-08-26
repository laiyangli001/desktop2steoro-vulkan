"""Explicit DirectML resource-sharing decisions.

This module does not infer zero-copy from a Python object type. A native
resource is eligible for DirectML only after adapter identity, shape, format,
ownership, and an import/copy operation are all established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app_runtime.interop import validate_resource_share


DirectMLResourceMode = Literal["shared", "gpu_copy", "cpu_compat", "rejected"]


@dataclass(frozen=True)
class DirectMLResourceDecision:
    mode: DirectMLResourceMode
    allowed: bool
    reason: str
    adapter_luid: int = 0
    consumer_adapter_luid: int = 0
    gpu_to_cpu: bool = False
    gpu_copy_count: int = 0
    zero_copy: bool = False
    zero_copy_ready: bool = False

    def to_report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "reason": self.reason,
            "adapter_luid": self.adapter_luid,
            "consumer_adapter_luid": self.consumer_adapter_luid,
            "gpu_to_cpu": self.gpu_to_cpu,
            "gpu_copy_count": self.gpu_copy_count,
            "zero_copy": self.zero_copy,
            "zero_copy_ready": self.zero_copy_ready,
        }


def _resource_luid(resource: Any) -> int:
    return int(getattr(resource, "adapter_luid", 0) or 0)


def _has_shared_handle(resource: Any) -> bool:
    for name in ("shared_handle", "d3d12_shared_handle", "directml_shared_handle"):
        try:
            if int(getattr(resource, name, 0) or 0) != 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _has_gpu_copy_bridge(resource: Any) -> bool:
    return any(
        callable(getattr(resource, name, None))
        for name in ("copy_to_directml", "copy_to_d3d12", "copy_to_shared_resource")
    )


def assess_directml_resource(
    resource: Any,
    *,
    consumer_adapter_luid: int | None,
    expected_format: str | None = "BGRA8",
    expected_width: int | None = None,
    expected_height: int | None = None,
    allow_cpu_fallback: bool = True,
) -> DirectMLResourceDecision:
    """Choose shared, one-GPU-copy, explicit CPU compatibility, or reject.

    CPU compatibility is reported as a distinct mode. It is never represented
    as zero_copy=True and callers can disable it to fail fast.
    """
    producer_luid = _resource_luid(resource)
    consumer_luid = int(consumer_adapter_luid or 0)
    shape_decision = validate_resource_share(
        resource,
        consumer_luid,
        expected_format=expected_format,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    if not shape_decision.allowed:
        if allow_cpu_fallback:
            return DirectMLResourceDecision(
                "cpu_compat",
                True,
                f"DirectML native share rejected: {shape_decision.reason}; "
                "using explicit CPU compatibility input",
                producer_luid,
                consumer_luid,
                gpu_to_cpu=True,
                gpu_copy_count=1,
            )
        return DirectMLResourceDecision(
            "rejected",
            False,
            shape_decision.reason,
            producer_luid,
            consumer_luid,
        )

    if _has_shared_handle(resource):
        return DirectMLResourceDecision(
            "shared",
            True,
            "D3D11/D3D12 shared handle and matching Adapter LUID",
            producer_luid,
            consumer_luid,
            gpu_to_cpu=False,
            gpu_copy_count=0,
            zero_copy=False,
            zero_copy_ready=False,
        )

    if _has_gpu_copy_bridge(resource):
        return DirectMLResourceDecision(
            "gpu_copy",
            True,
            "matching Adapter LUID; one GPU-internal copy bridge is available",
            producer_luid,
            consumer_luid,
            gpu_to_cpu=False,
            gpu_copy_count=1,
        )

    if allow_cpu_fallback:
        return DirectMLResourceDecision(
            "cpu_compat",
            True,
            "resource identity matches but no DirectML import/copy bridge is exposed",
            producer_luid,
            consumer_luid,
            gpu_to_cpu=True,
            gpu_copy_count=1,
        )

    return DirectMLResourceDecision(
        "rejected",
        False,
        "resource identity matches but no DirectML import/copy bridge is exposed",
        producer_luid,
        consumer_luid,
    )


def probe_directml_resource_capabilities() -> dict[str, Any]:
    """Report implementation presence without claiming hardware validation."""
    try:
        from viewer.vulkan_resources import VulkanD3D11ImportedImage
    except Exception as exc:
        return {
            "d3d11_shared_resource": {
                "implementation_present": False,
                "hardware_verified": False,
                "status": "待验证",
                "reason": f"{type(exc).__name__}: {exc}",
            },
            "vulkan_external_memory": {
                "implementation_present": False,
                "hardware_verified": False,
                "status": "待验证",
                "reason": "Vulkan D3D11 import class unavailable",
            },
        }

    return {
        "d3d11_shared_resource": {
            "implementation_present": True,
            "hardware_verified": False,
            "status": "待真实硬件验证",
            "entrypoint": "VulkanD3D11ImportedImage",
            "resource_contract": "shared_handle + matching Adapter LUID + BGRA8 dimensions",
        },
        "vulkan_external_memory": {
            "implementation_present": bool(VulkanD3D11ImportedImage),
            "hardware_verified": False,
            "status": "待真实硬件验证",
            "handle_type": "VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT",
            "zero_copy": False,
        },
    }


__all__ = [
    "DirectMLResourceDecision",
    "DirectMLResourceMode",
    "assess_directml_resource",
    "probe_directml_resource_capabilities",
]
