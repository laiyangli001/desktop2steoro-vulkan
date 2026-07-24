"""Consume runtime results without introducing a CPU image round trip."""

from __future__ import annotations

import os
import queue
import threading
import time

from viewer.cuda_vulkan_interop import CudaVulkanImageImporter
from viewer.rocm_vulkan_interop import RocmVulkanImageImporter
from viewer.vulkan_resources import (
    VulkanBinarySemaphore,
    VulkanExportableImage,
    VulkanExportableSemaphore,
    VulkanImageResource,
)

from .gpu_producer import GpuProducerAdapter, register_gpu_producer_adapter
from .output_contract import VulkanStereoOutputFrame


class CudaVulkanOutputAdapter(GpuProducerAdapter):
    """Convert CUDA RGBA tensors into persistent Vulkan image slots."""

    backend_name = "cuda"

    def _create_importer(self):
        return CudaVulkanImageImporter()

    @staticmethod
    def _source_image_contract(resource: VulkanImageResource) -> dict[str, object]:
        """Compatibility alias for callers using the former CUDA adapter API."""
        return GpuProducerAdapter.source_image_contract(resource)

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
        self.left_release_semaphores = []
        self.right_release_semaphores = []
        self.left_visible_semaphores = []
        self.right_visible_semaphores = []
        self.external_semaphore_enabled = False
        self.left_slot = None
        self.right_slot = None
        self._extent = None
        self._lease_condition = threading.Condition()
        self._active_leases: dict[int, int] = {}
        self._closed = False
        self._screen_light_rgb = (0.18, 0.18, 0.18)
        self._screen_light_pending = None
        self._release_signaled: set[tuple[int, int]] = set()
        self._source_frames: dict[int, tuple[object, object, int]] = {}
        self._released_source_frames: set[int] = set()
        self._prepared_source_eyes: set[tuple[int, int]] = set()


        self._screen_light_last_submit = 0.0

    def _update_screen_light_sample(self, left, right) -> None:
        """Asynchronously reduce display sRGB eyes to one linear screen color."""
        pending = self._screen_light_pending
        if pending is not None:
            host, event = pending
            if event.query():
                values = host.tolist()
                self._screen_light_rgb = tuple(
                    max(0.0, min(8.0, float(value))) for value in values[:3]
                )
                self._screen_light_pending = None

        now = time.monotonic()
        if (
            self._screen_light_pending is not None
            or now - self._screen_light_last_submit < 0.25
        ):
            return
        try:
            import torch

            if (
                not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or not left.is_cuda
                or not right.is_cuda
                or left.ndim != 3
                or right.ndim != 3
                or left.shape[-1] < 3
                or right.shape[-1] < 3
            ):
                return
            row_step = max(1, int(left.shape[0]) // 32)
            column_step = max(1, int(left.shape[1]) // 32)
            sampled = torch.cat(
                (
                    left[::row_step, ::column_step, :3].reshape(-1, 3),
                    right[::row_step, ::column_step, :3].reshape(-1, 3),
                ),
                dim=0,
            ).to(dtype=torch.float32)
            if left.dtype == torch.uint8:
                sampled = sampled / 255.0
            sampled = sampled.clamp(0.0, 1.0)
            linear = torch.where(
                sampled <= 0.04045,
                sampled / 12.92,
                ((sampled + 0.055) / 1.055).pow(2.4),
            )
            rgb = linear.mean(dim=0)
            host = torch.empty(3, dtype=torch.float32, pin_memory=True)
            host.copy_(rgb, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(left.device))
            self._screen_light_pending = (host, event)
            self._screen_light_last_submit = now

        except Exception:
            # Screen illumination is supplemental and must never break output.
            self._screen_light_pending = None

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
                    if not any(
                        prepared_frame == int(frame_id)
                        for prepared_frame, _eye in getattr(
                            self, "_prepared_source_eyes", ()
                        )
                    ):
                        getattr(self, "_source_frames", {}).pop(int(frame_id), None)
                    return

    def prepare_source_for_sampling(self, frame_id: int, eye_index: int):
        """Wait for the producer and publish a post-barrier semaphore to Filament."""
        entry = self._source_frames.get(int(frame_id))
        if entry is None:
            raise RuntimeError(f"unknown Vulkan source frame {frame_id}")
        left, right, slot_index = entry
        resource = left if int(eye_index) == 0 else right
        ready = (
            self.left_ready_semaphores[slot_index]
            if int(eye_index) == 0
            else self.right_ready_semaphores[slot_index]
        )
        visible = (
            self.left_visible_semaphores[slot_index]
            if int(eye_index) == 0
            else self.right_visible_semaphores[slot_index]
        )
        self.presenter.vulkan.prepare_external_image_for_sampling(
            resource.resource,
            wait_semaphore=ready.semaphore,
            signal_semaphore=visible.semaphore,
        )
        self._prepared_source_eyes.add((int(frame_id), int(eye_index)))
        return visible.semaphore

    def release_consumer_frame(
        self, frame_id: int, consumer_semaphores=None
    ) -> None:
        """Signal producer-release semaphores after Filament has finished sampling."""
        frame_key = int(frame_id)
        if frame_key in self._released_source_frames:
            return
        entry = self._source_frames.get(frame_key)
        if entry is None:
            self.release_frame(frame_key)
            self._released_source_frames.add(frame_key)
            return
        left, right, slot_index = entry
        waits = tuple(consumer_semaphores or ())
        for eye_index, resource, release_semaphore in (
            (0, left.resource, self.left_release_semaphores[slot_index]),
            (1, right.resource, self.right_release_semaphores[slot_index]),
        ):
            if (frame_key, eye_index) not in self._prepared_source_eyes:
                continue
            self.presenter.vulkan.release_external_image_from_sampling(
                resource,
                wait_semaphore=(
                    waits[eye_index] if eye_index < len(waits) else None
                ),
                signal_semaphore=release_semaphore.semaphore,
            )
            self._release_signaled.add((eye_index, slot_index))
            self._prepared_source_eyes.discard((frame_key, eye_index))
        self.release_frame(frame_key)
        self._released_source_frames.add(frame_key)
        self._source_frames.pop(frame_key, None)

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
        self.importer = self._create_importer()
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
            and getattr(self.presenter, "screen_ready_semaphore_available", False)
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
            self.left_release_semaphores = [
                VulkanExportableSemaphore(context, label=f"runtime-left-release-{index}")
                for index in range(self.ring_size)
            ]
            self.right_release_semaphores = [
                VulkanExportableSemaphore(context, label=f"runtime-right-release-{index}")
                for index in range(self.ring_size)
            ]
            self.left_visible_semaphores = [
                VulkanBinarySemaphore(context, label=f"runtime-left-visible-{index}")
                for index in range(self.ring_size)
            ]
            self.right_visible_semaphores = [
                VulkanBinarySemaphore(context, label=f"runtime-right-visible-{index}")
                for index in range(self.ring_size)
            ]
            for semaphore in (
                *self.left_ready_semaphores,
                *self.right_ready_semaphores,
                *self.left_release_semaphores,
                *self.right_release_semaphores,
            ):
                self.importer.register_semaphore(semaphore)
            self.external_semaphore_enabled = self.importer.capabilities.external_semaphore
        except Exception:
            for semaphore in (
                *self.left_ready_semaphores,
                *self.right_ready_semaphores,
                *self.left_release_semaphores,
                *self.right_release_semaphores,
                *self.left_visible_semaphores,
                *self.right_visible_semaphores,
            ):
                semaphore.close()
            self.left_ready_semaphores = []
            self.right_ready_semaphores = []
            self.left_release_semaphores = []
            self.right_release_semaphores = []
            self.left_visible_semaphores = []
            self.right_visible_semaphores = []
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
            self._update_screen_light_sample(left, right)
            left_ready = None
            right_ready = None
            use_external_semaphore = bool(
                self.external_semaphore_enabled
                and getattr(self.presenter, "screen_ready_semaphore_available", False)
            )
            if use_external_semaphore:
                for eye_index, release_semaphore in (
                    (0, self.left_release_semaphores[slot_index]),
                    (1, self.right_release_semaphores[slot_index]),
                ):
                    if (eye_index, slot_index) not in self._release_signaled:
                        continue
                    self.importer.wait_semaphore(release_semaphore)
                    self._release_signaled.discard((eye_index, slot_index))
                self.importer.copy_tensor(left, self.left_slot)
                self.importer.copy_tensor(right, self.right_slot)
                left_ready = self.left_ready_semaphores[slot_index]
                right_ready = self.right_ready_semaphores[slot_index]
                stream = None
                self.importer.signal_semaphore(left_ready, stream=stream)
                self.importer.signal_semaphore(right_ready, stream=stream)
            if not use_external_semaphore:
                self.importer.copy_tensor(left, self.left_slot)
                self.importer.copy_tensor(right, self.right_slot)
                self.importer.synchronize()
        except Exception:
            self.release_frame(frame_id)
            raise
        left_contract = self.source_image_contract(self.left_slot.resource)
        right_contract = self.source_image_contract(self.right_slot.resource)
        self._source_frames[int(frame_id)] = (
            self.left_slot,
            self.right_slot,
            slot_index,
        )
        self._released_source_frames.discard(int(frame_id))
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
                    self.external_semaphore_sync_mode
                    if use_external_semaphore
                    else self.output_sync_mode
                ),
                "vulkan_ready_semaphore_left": (
                    left_ready.semaphore if left_ready is not None else None
                ),
                "vulkan_ready_semaphore_right": (
                    right_ready.semaphore if right_ready is not None else None
                ),
                "vulkan_source_layout_left": left_contract["layout"],
                "vulkan_source_layout_right": right_contract["layout"],
                "vulkan_source_queue_family_left": left_contract["queue_family"],
                "vulkan_source_queue_family_right": right_contract["queue_family"],
                "_vulkan_source_prepare_for_sampling": self.prepare_source_for_sampling,
                "_vulkan_source_consumer_release": self.release_consumer_frame,
                "_vulkan_output_release": self.release_frame,
                "screen_light_linear_rgb": self._screen_light_rgb,
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
        for semaphore in (
            *self.left_ready_semaphores,
            *self.right_ready_semaphores,
            *self.left_release_semaphores,
            *self.right_release_semaphores,
            *self.left_visible_semaphores,
            *self.right_visible_semaphores,
        ):
            if semaphore is not None:
                semaphore.close()
        self.left_ready_semaphores = []
        self.right_ready_semaphores = []
        self.left_release_semaphores = []
        self.right_release_semaphores = []
        self.left_visible_semaphores = []
        self.right_visible_semaphores = []
        self.external_semaphore_enabled = False
        self.left_slot = None
        self.right_slot = None
        self._extent = None
        self._screen_light_pending = None
        self._release_signaled.clear()
        self._source_frames.clear()
        self._released_source_frames.clear()
        self._prepared_source_eyes.clear()


class RocmVulkanOutputAdapter(CudaVulkanOutputAdapter):
    """Convert HIP tensors using the AMD ROCm Vulkan interop importer."""

    backend_name = "rocm"

    @staticmethod
    def _external_semaphore_requested() -> bool:
        value = os.environ.get("D2S_ENABLE_ROCM_EXTERNAL_SEMAPHORE", "auto")
        normalized = value.strip().lower()
        if normalized in {"", "auto", "default"}:
            return True
        return normalized in {"1", "true", "yes", "on"}

    def _create_importer(self):
        return RocmVulkanImageImporter()


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
            if self.sink is not None:
                ready = getattr(self.sink, "output_ready", None)
                if ready is None:
                    ready = getattr(self.sink, "initialized", True)
                if not bool(ready):
                    self.source_stat_inc("runtime_output_waiting_for_openxr")
                    self.shutdown_event.wait(0.01)
                    continue
            item = self._take_latest()
            if item is None:
                continue
            try:
                runtime_result, capture_timestamp = item
            except (TypeError, ValueError):
                self.source_stat_inc("runtime_output_invalid_item")
                continue
            submit_runtime_result = getattr(self.sink, "submit_runtime_result", None)
            if callable(submit_runtime_result):
                submit_runtime_result(
                    runtime_result,
                    float(capture_timestamp or time.monotonic()),
                )
                self.source_stat_inc("runtime_output_frames")
                continue
            frame = self._to_output_frame(item)
            if frame is None:
                continue
            if self.sink is None:
                self.source_stat_inc("runtime_output_no_sink")
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


register_gpu_producer_adapter("cuda", CudaVulkanOutputAdapter)
register_gpu_producer_adapter("nvidia", CudaVulkanOutputAdapter)
register_gpu_producer_adapter("rocm", RocmVulkanOutputAdapter)
register_gpu_producer_adapter("hip", RocmVulkanOutputAdapter)
