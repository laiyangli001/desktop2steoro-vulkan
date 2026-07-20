from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Protocol


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


ComputePassRecorder = Callable[[Any, VulkanStereoSubmission], None]


class VulkanComputeGraph:
    """Small executable scheduling shell for the first Vulkan compute pass."""

    def __init__(self, context: Any, record_pass: ComputePassRecorder) -> None:
        self._context = context
        self._record_pass = record_pass
        self._state = VulkanGraphState.CREATED
        self._latest: VulkanStereoSubmission | None = None
        self._last_timeline_value: int | None = None
        self._state = VulkanGraphState.READY

    @classmethod
    def from_pipeline(
        cls,
        context: Any,
        pipeline: Any,
        *,
        group_counts: tuple[int, int, int] = (1, 1, 1),
        descriptor_set: Any | None = None,
    ) -> "VulkanComputeGraph":
        if len(group_counts) != 3 or any(int(value) < 1 for value in group_counts):
            raise ValueError("group_counts must contain three positive values")

        def record_pass(command_buffer: Any, _submission: VulkanStereoSubmission) -> None:
            counts = {
                "group_count_x": int(group_counts[0]),
                "group_count_y": int(group_counts[1]),
                "group_count_z": int(group_counts[2]),
            }
            if descriptor_set is not None:
                counts["descriptor_set"] = descriptor_set
            pipeline.record_dispatch(command_buffer, **counts)

        return cls(context, record_pass)

    @property
    def state(self) -> VulkanGraphState:
        return self._state

    @property
    def last_timeline_value(self) -> int | None:
        return self._last_timeline_value

    def enqueue(self, submission: VulkanStereoSubmission) -> None:
        if self._state is not VulkanGraphState.READY:
            raise RuntimeError(f"Vulkan graph is not ready: {self._state.value}")
        self._latest = submission

    def flush(self) -> int | None:
        if self._state is not VulkanGraphState.READY:
            raise RuntimeError(f"Vulkan graph is not ready: {self._state.value}")
        submission = self._latest
        self._latest = None
        if submission is None:
            return None
        self._last_timeline_value = self._context.submit_on(
            "compute",
            lambda command_buffer: self._record_pass(command_buffer, submission),
        )
        return self._last_timeline_value

    def submit(self, submission: VulkanStereoSubmission) -> int | None:
        self.enqueue(submission)
        return self.flush()

    def close(self) -> None:
        self._latest = None
        self._state = VulkanGraphState.CLOSED

