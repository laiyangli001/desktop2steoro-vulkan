from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class VulkanGraphState(StrEnum):
    CREATED = "created"
    READY = "ready"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class VulkanStereoSubmission:
    frame_id: int
    rgb_handle: object
    depth_handle: object
    config_version: int


class VulkanStereoGraph(Protocol):
    @property
    def state(self) -> VulkanGraphState: ...

    def submit(self, submission: VulkanStereoSubmission) -> object: ...

    def close(self) -> None: ...

