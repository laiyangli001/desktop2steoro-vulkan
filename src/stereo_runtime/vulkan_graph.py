from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable, Protocol


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
    ready_timeline: int | None = None


class VulkanStereoGraph(Protocol):
    @property
    def state(self) -> VulkanGraphState: ...

    def submit(self, submission: VulkanStereoSubmission) -> object: ...

    def close(self) -> None: ...


ComputePassRecorder = Callable[[Any, VulkanStereoSubmission], None]


@dataclass(frozen=True, slots=True)
class VulkanPassDeclaration:
    name: str
    group_counts: tuple[int, int, int] = (1, 1, 1)
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Vulkan pass name must not be empty")
        if len(self.group_counts) != 3 or any(int(value) < 1 for value in self.group_counts):
            raise ValueError("Vulkan pass group_counts must contain three positive values")
        if any(not resource.strip() for resource in (*self.reads, *self.writes)):
            raise ValueError("Vulkan pass resources must not be empty")


@dataclass(frozen=True, slots=True)
class VulkanComputePass:
    declaration: VulkanPassDeclaration
    pipeline: Any
    descriptor_set: Any | None = None

    def record(self, command_buffer: Any) -> None:
        counts = {
            "group_count_x": int(self.declaration.group_counts[0]),
            "group_count_y": int(self.declaration.group_counts[1]),
            "group_count_z": int(self.declaration.group_counts[2]),
        }
        if self.descriptor_set is not None:
            counts["descriptor_set"] = self.descriptor_set
        self.pipeline.record_dispatch(command_buffer, **counts)


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

    @classmethod
    def from_passes(
        cls,
        context: Any,
        passes: Iterable[VulkanComputePass],
    ) -> "VulkanComputeGraph":
        declared = tuple(passes)
        if not declared:
            raise ValueError("Vulkan compute graph requires at least one pass")
        names = [item.declaration.name for item in declared]
        if len(names) != len(set(names)):
            raise ValueError("Vulkan compute pass names must be unique")

        def record_pass(command_buffer: Any, _submission: VulkanStereoSubmission) -> None:
            previous_writes: set[str] = set()
            for index, compute_pass in enumerate(declared):
                current = compute_pass.declaration
                dependency_resources = previous_writes.intersection(
                    {*current.reads, *current.writes}
                )
                if dependency_resources:
                    _record_compute_memory_barrier(context.vk, command_buffer)
                compute_pass.record(command_buffer)
                previous_writes = set(current.writes)

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
        submit_kwargs = {}
        if submission.ready_timeline is not None:
            submit_kwargs["wait_for_timeline"] = submission.ready_timeline
        self._last_timeline_value = self._context.submit_on(
            "compute",
            lambda command_buffer: self._record_pass(command_buffer, submission),
            **submit_kwargs,
        )
        return self._last_timeline_value

    def submit(self, submission: VulkanStereoSubmission) -> int | None:
        self.enqueue(submission)
        return self.flush()

    def close(self) -> None:
        self._latest = None
        self._state = VulkanGraphState.CLOSED


def _record_compute_memory_barrier(vk: Any, command_buffer: Any) -> None:
    barrier = vk.VkMemoryBarrier(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
    )
    vk.vkCmdPipelineBarrier(
        command_buffer,
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        [barrier],
        0,
        None,
        0,
        None,
    )

