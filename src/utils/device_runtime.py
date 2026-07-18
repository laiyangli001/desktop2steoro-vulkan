from __future__ import annotations

from dataclasses import dataclass

from .device import get_device


@dataclass(frozen=True)
class DeviceRuntime:
    device: object
    device_info: str


def resolve_device_runtime(device_id: int) -> DeviceRuntime:
    device, device_info = get_device(device_id)
    return DeviceRuntime(device=device, device_info=device_info)
