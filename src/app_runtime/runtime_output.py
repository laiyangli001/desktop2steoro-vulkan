"""Consume runtime results without introducing a CPU image round trip."""

from __future__ import annotations

import os
import queue
import threading
import time

from viewer.cuda_vulkan_interop import CudaVulkanImageImporter
from viewer.vulkan_resources import (
    VulkanExportableImage,
    VulkanExportableSemaphore,
    VulkanImageResource,
)

from .output_contract import VulkanStereoOutputFrame


class CudaVulkanOutputAdapter:
    """Convert CUDA RGBA tensors into persistent Vulkan image slots."""

    @staticmethod
    def _external_semaphore_requested() -> bool:
        value = os.environ.get("D2S_ENABLE_CUDA_EXTERNAL_SEMAPHORE", "0")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def __init__(self, presenter):
        self.presenter = presenter
        self.importer = None
        self.ring_size = max(2, int(os.environ.get("D2S_VULKAN_OUTPUT_RING_SIZE", "3")))
        self.left_slots = []
        self.right_slots = []
        self.left_ready_semaphores = []
        self.right_ready_semaphores = []
        self.external_semaphore_enabled = False
        self.left_slot = None
        self.right_slot = None
        self._extent = None
        self._lease_condition = threading.Condition()
        self._active_leases: dict[int, int] = {}
        self._closed = False

    @staticmethod
    def _tensor_extent(tensor):
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        if len(shape) != 3 or shape[-1] != 4:
            raise ValueError("runtime eye must be an HxWx4 tensor")
        return shape[1], shape[0]

    def _claim_slot(self, slot_index: int, frame_id: int) -> None:
        with self._lease_condition:
            while slot_index in self._active_leases and not self._closed:
                self._lease_condition.wait()
            if self._closed:
                raise RuntimeError("Vulkan output adapter is closed")
            self._active_leases[slot_index] = int(frame_id)

    def release_frame(self, frame_id: int) -> None:
        """Release a producer slot after the XR consumer no longer samples it."""
        with self._lease_condition:
            for slot_index, lease_frame_id in tuple(self._active_leases.items()):
                if lease_frame_id == int(frame_id):
                    del self._active_leases[slot_index]
                    self._lease_condition.notify_all()
                    return

    def _ensure_slots(self, width: int, height: int) -> None:
        if not bool(getattr(self.presenter, "initialized", False)):
            raise RuntimeError("OpenXR Vulkan presenter is not initialized")
        context = self.presenter.vulkan
        if context is None:
            raise RuntimeError("OpenXR Vulkan context is unavailable")
        if self._extent == (width, height):
            return
        self.close()
        with self._lease_condition:
            self._closed = False
        self.importer = CudaVulkanImageImporter()
        # Runtime eye tensors contain display-referred sRGB bytes. Keep the
        # Vulkan image format sRGB so Quad Layer copies remain byte-preserving.
        output_format = context.vk.VK_FORMAT_R8G8B8A8_SRGB
        self.left_slots = [
            VulkanExportableImage(
                context, width, height, label=f"runtime-left-eye-{index}", format=output_format
            )
            for index in range(self.ring_size)
        ]
        self.right_slots = [
            VulkanExportableImage(
                context, width, height, label=f"runtime-right-eye-{index}", format=output_format
            )
            for index in range(self.ring_size)
        ]
        external_semaphore_requested = bool(
            self._external_semaphore_requested()
            and getattr(
                getattr(self.presenter, "filament_bridge", None),
                "screen_ready_semaphore_abi_available",
                False,
            )
        )
        try:
            if not external_semaphore_requested:
                self._extent = (width, height)
                return
            self.left_ready_semaphores = [
                VulkanExportableSemaphore(
                    context, label=f"runtime-left-ready-{index}"
                )
                for index in range(self.ring_size)
            ]
            self.right_ready_semaphores = [
                VulkanExportableSemaphore(
                    context, label=f"runtime-right-ready-{index}"
                )
                for index in range(self.ring_size)
            ]
            for semaphore in (*self.left_ready_semaphores, *self.right_ready_semaphores):
                self.importer.register_semaphore(semaphore)
            self.external_semaphore_enabled = self.importer.capabilities.external_semaphore
        except Exception:
            for semaphore in (*self.left_ready_semaphores, *self.right_ready_semaphores):
                semaphore.close()
            self.left_ready_semaphores = []
            self.right_ready_semaphores = []
            self.external_semaphore_enabled = False
        self._extent = (width, height)

    def convert(self, runtime_result, *, frame_id: int, timestamp: float):
        left = getattr(runtime_result, "left_eye", None)
        right = getattr(runtime_result, "right_eye", None)
        width, height = self._tensor_extent(left)
        if self._tensor_extent(right) != (width, height):
            raise ValueError("left/right runtime eye dimensions differ")
        self._ensure_slots(width, height)
        slot_index = int(frame_id) % self.ring_size
        self._claim_slot(slot_index, frame_id)
        self.left_slot = self.left_slots[slot_index]
        self.right_slot = self.right_slots[slot_index]
        try:
            self.importer.copy_tensor(left, self.left_slot)
            self.importer.copy_tensor(right, self.right_slot)
            left_ready = None
            right_ready = None
            bridge = getattr(self.presenter, "filament_bridge", None)
            use_external_semaphore = bool(
                self.external_semaphore_enabled
                and getattr(bridge, "screen_ready_semaphore_abi_available", False)
            )
            if use_external_semaphore:
                left_ready = self.left_ready_semaphores[slot_index]
                right_ready = self.right_ready_semaphores[slot_index]
                stream = None
                self.importer.signal_semaphore(left_ready, stream=stream)
                self.importer.signal_semaphore(right_ready, stream=stream)
            if not use_external_semaphore:
                self.importer.synchronize()
        except Exception:
            self.release_frame(frame_id)
            raise
        return VulkanStereoOutputFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            left_eye=self.left_slot.resource,
            right_eye=self.right_slot.resource,
            ready_timeline=None,
            metadata={
                **dict(getattr(runtime_result, "debug_info", None) or {}),
                "vulkan_output_ring_slot": slot_index,
                "vulkan_output_ring_size": self.ring_size,
                "vulkan_output_sync": (
                    "cuda_external_semaphore"
                    if use_external_semaphore
                    else "cuda_stream_synchronized"
                ),
                "vulkan_ready_semaphore_left": (
                    left_ready.semaphore if left_ready is not None else None
                ),
                "vulkan_ready_semaphore_right": (
                    right_ready.semaphore if right_ready is not None else None
                ),
                "_vulkan_output_release": self.release_frame,
            },
            color_space="srgb",
            image_origin="top_left",
        )

    def close(self) -> None:
        with self._lease_condition:
            self._closed = True
            self._active_leases.clear()
            self._lease_condition.notify_all()
        if self.importer is not None:
            self.importer.close()
        self.importer = None
        for slot in (*self.left_slots, *self.right_slots):
            if slot is not None:
                slot.close()
        self.left_slots = []
        self.right_slots = []
        for semaphore in (*self.left_ready_semaphores, *self.right_ready_semaphores):
            if semaphore is not None:
                semaphore.close()
        self.left_ready_semaphores = []
        self.right_ready_semaphores = []
        self.external_semaphore_enabled = False
        self.left_slot = None
        self.right_slot = None
        self._extent = None


class VulkanRuntimeOutputConsumer:
    """Bridge the bounded runtime queue to a Vulkan-capable output sink."""

    def __init__(self, *, runtime_q, shutdown_event, source_stat_inc, sink=None, gpu_adapter=None):
        self.runtime_q = runtime_q
        self.shutdown_event = shutdown_event
        self.source_stat_inc = source_stat_inc
        self.sink = sink
        self.gpu_adapter = gpu_adapter
        self._next_frame_id = 0

    def _take_latest(self):
        try:
            item = self.runtime_q.get(timeout=0.05)
        except queue.Empty:
            return None
        while True:
            try:
                item = self.runtime_q.get_nowait()
                self.source_stat_inc("runtime_output_overwrite")
            except queue.Empty:
                return item

    def _to_output_frame(self, item):
        try:
            runtime_result, capture_timestamp = item
        except (TypeError, ValueError):
            self.source_stat_inc("runtime_output_invalid_item")
            return None

        left_eye = getattr(runtime_result, "left_eye", None)
        right_eye = getattr(runtime_result, "right_eye", None)
        if not isinstance(left_eye, VulkanImageResource) or not isinstance(
            right_eye, VulkanImageResource
        ):
            if self.gpu_adapter is None:
                # Torch/CPU results wait for a vendor interop importer; never copy them here.
                self.source_stat_inc("runtime_output_waiting_for_vulkan_importer")
                return None
            try:
                frame = self.gpu_adapter.convert(
                    runtime_result,
                    frame_id=self._next_frame_id,
                    timestamp=float(capture_timestamp or time.monotonic()),
                )
            except Exception as exc:
                self.source_stat_inc(
                    "runtime_output_import_errors",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                return None
            self._next_frame_id += 1
            self.source_stat_inc("runtime_output_gpu_copies")
            return frame

        frame = VulkanStereoOutputFrame(
            frame_id=self._next_frame_id,
            timestamp=float(capture_timestamp or time.monotonic()),
            left_eye=left_eye,
            right_eye=right_eye,
            sbs=getattr(runtime_result, "sbs", None),
            ready_timeline=getattr(runtime_result, "ready_timeline", None),
            metadata=dict(getattr(runtime_result, "debug_info", None) or {}),
            color_space=str(
                (getattr(runtime_result, "debug_info", None) or {}).get(
                    "output_color_space", "srgb"
                )
            ),
            image_origin=str(
                (getattr(runtime_result, "debug_info", None) or {}).get(
                    "output_image_origin", "top_left"
                )
            ),
        )
        self._next_frame_id += 1
        return frame

    def _release_frame(self, frame) -> None:
        release = getattr(self.gpu_adapter, "release_frame", None)
        if callable(release):
            release(frame.frame_id)

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            item = self._take_latest()
            if item is None:
                continue
            frame = self._to_output_frame(item)
            if frame is None:
                continue
            if self.sink is None:
                self.source_stat_inc("runtime_output_no_sink")
                self._release_frame(frame)
                continue
            if not bool(getattr(self.sink, "initialized", True)):
                self.source_stat_inc("runtime_output_waiting_for_openxr")
                self._release_frame(frame)
                continue
            try:
                self.sink.submit_output(frame)
            except Exception as exc:
                self._release_frame(frame)
                self.source_stat_inc(
                    "runtime_output_submit_errors",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
            else:
                self.source_stat_inc("runtime_output_frames")

    def close(self) -> None:
        close = getattr(self.gpu_adapter, "close", None)
        if callable(close):
            close()
