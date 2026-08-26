"""Backend-neutral resource identity and sharing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterInteropDecision:
    allowed: bool
    reason: str
    producer_luid: int = 0
    consumer_luid: int = 0
    gpu_copy_count: int = 0
    zero_copy: bool = False

    def to_report(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "producer_adapter_luid": self.producer_luid,
            "consumer_adapter_luid": self.consumer_luid,
            "gpu_copy_count": self.gpu_copy_count,
            "zero_copy": self.zero_copy,
        }


def validate_adapter_luid(
    producer_luid: int | None,
    consumer_luid: int | None,
    *,
    require_match: bool = True,
) -> AdapterInteropDecision:
    """Reject missing or mismatched DXGI/Vulkan adapter identities."""
    producer = int(producer_luid or 0)
    consumer = int(consumer_luid or 0)
    if require_match and producer == 0:
        return AdapterInteropDecision(False, "producer Adapter LUID is missing", producer, consumer)
    if require_match and consumer == 0:
        return AdapterInteropDecision(False, "consumer Adapter LUID is missing", producer, consumer)
    if require_match and producer != consumer:
        return AdapterInteropDecision(
            False,
            f"Adapter LUID mismatch: producer=0x{producer:016x}, consumer=0x{consumer:016x}",
            producer,
            consumer,
        )
    return AdapterInteropDecision(
        True,
        "matching Adapter LUID",
        producer,
        consumer,
        gpu_copy_count=0,
        zero_copy=True,
    )


def validate_resource_share(
    resource: Any,
    consumer_luid: int | None,
    *,
    expected_format: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> AdapterInteropDecision:
    """Validate identity and basic shape before attempting an external import."""
    producer_luid = int(getattr(resource, "adapter_luid", 0) or 0)
    decision = validate_adapter_luid(producer_luid, consumer_luid)
    if not decision.allowed:
        return decision
    resource_format = str(getattr(resource, "format", "") or "")
    if expected_format and resource_format and resource_format.lower() != expected_format.lower():
        return AdapterInteropDecision(
            False,
            f"resource format mismatch: producer={resource_format}, expected={expected_format}",
            producer_luid,
            int(consumer_luid or 0),
        )
    for name, expected in (("width", expected_width), ("height", expected_height)):
        if expected is None:
            continue
        actual = int(getattr(resource, name, 0) or 0)
        if actual != int(expected):
            return AdapterInteropDecision(
                False,
                f"resource {name} mismatch: producer={actual}, expected={int(expected)}",
                producer_luid,
                int(consumer_luid or 0),
            )
    return decision


__all__ = [
    "AdapterInteropDecision",
    "validate_adapter_luid",
    "validate_resource_share",
]
