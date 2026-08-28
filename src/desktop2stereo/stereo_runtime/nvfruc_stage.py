"""Ordered process-local NvFRUC stage for synchronized stereo output.

The stage consumes completed stereo runtime results and publishes complete frame
groups in presentation order: real A, generated midpoint, real B.  It keeps
all image conversion on CUDA through NvFrucStereoGenerator.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from typing import Any, Callable

from .nvfruc import NvFrucStereoGenerator, NvFrucUnavailable
from .output import make_sbs


class NvFrucStage:
    """Bridge the latest runtime queue to an ordered presentation queue."""

    def __init__(
        self,
        *,
        input_q: Any,
        output_q: Any,
        shutdown_event: Any,
        output_format_provider: Callable[[], str] | None = None,
        device_index: int = 0,
        cuda_stream: int = 0,
        calibration_controller: Any | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.input_q = input_q
        self.output_q = output_q
        self.shutdown_event = shutdown_event
        self.output_format_provider = output_format_provider
        self.device_index = int(device_index)
        self.cuda_stream = int(cuda_stream)
        self.calibration_controller = calibration_controller
        self.on_status = on_status
        self._generator: NvFrucStereoGenerator | None = None
        self._previous: tuple[Any, float] | None = None
        self._stopped = False
        self._real_frames = 0
        self._generated_frames = 0
        self._group_drops = 0
        self._resets = 0
        self._interpolate_seconds = 0.0
        self._last_report = time.perf_counter()
        if self.calibration_controller is not None:
            self.calibration_controller.start()

    def _status(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)

    @staticmethod
    def _timestamp(item: tuple[Any, Any]) -> float:
        try:
            return float(item[1])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _make_generated_result(
        self,
        previous_result: Any,
        generated_left: Any,
        generated_right: Any,
        timestamp: float,
    ) -> Any:
        output_format = str(
            self.output_format_provider() if self.output_format_provider else
            getattr(previous_result, "output_format", "half_sbs")
        ).strip().lower() or "half_sbs"
        debug = dict(getattr(previous_result, "debug_info", {}) or {})
        debug["nvfruc_generated"] = True
        debug["nvfruc_timestamp"] = timestamp
        generated_sbs = getattr(previous_result, "sbs", None)
        if output_format not in {"openxr_full_synthesis_eyes", "openxr"}:
            generated_sbs = make_sbs(
                generated_left,
                generated_right,
                output_format,
                fused=False,
            )
        return replace(
            previous_result,
            left_eye=generated_left,
            right_eye=generated_right,
            sbs=generated_sbs,
            debug_info=debug,
            native_final_sbs_surface=None,
            vulkan_compute_request=None,
        )

    def _put_group(self, items: list[tuple[Any, float]]) -> None:
        """Publish one complete A/midpoint/B group without partial eviction."""
        capacity = int(getattr(self.output_q, "maxsize", 0) or 0)
        while not self.shutdown_event.is_set():
            if capacity and self.output_q.qsize() + len(items) > capacity:
                # The stage is the only producer; remove whole groups from the
                # oldest side before publishing the next complete group.
                queued = self.output_q.qsize()
                if queued < 3:
                    # Wait for the consumer to finish a partially consumed
                    # group; never discard only one or two group members.
                    time.sleep(0.005)
                    continue
                remove_count = 3
                self._group_drops += 1
                for _ in range(remove_count):
                    try:
                        self.output_q.get_nowait()
                    except queue.Empty:
                        break
                continue
            try:
                for item in items:
                    self.output_q.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def _ensure_generator(self, result: Any) -> NvFrucStereoGenerator:
        if self._generator is None:
            left = getattr(result, "left_eye", None)
            right = getattr(result, "right_eye", None)
            if left is None or right is None:
                raise NvFrucUnavailable(
                    "NvFRUC requires left_eye and right_eye runtime tensors"
                )
            self._generator = NvFrucStereoGenerator(
                left,
                right,
                device_index=self.device_index,
                cuda_stream=self.cuda_stream,
            )
            self._status("NvFRUC native stereo generator initialized")
        return self._generator

    def _publish_group(self, previous: tuple[Any, float], current: tuple[Any, float]) -> None:
        previous_result, previous_ts = previous
        current_result, current_ts = current
        generator = self._ensure_generator(current_result)
        interpolate_started = time.perf_counter()
        generated_left, generated_right = generator.interpolate(
            (getattr(previous_result, "left_eye"), getattr(previous_result, "right_eye")),
            (getattr(current_result, "left_eye"), getattr(current_result, "right_eye")),
            previous_timestamp=previous_ts,
            next_timestamp=current_ts,
            output_timestamp=(previous_ts + current_ts) * 0.5,
        )
        interpolate_seconds = time.perf_counter() - interpolate_started
        self._interpolate_seconds += interpolate_seconds
        if self.calibration_controller is not None:
            calibration_result = self.calibration_controller.observe(
                process_ms=interpolate_seconds * 1000.0,
                submitted_fps=0.0,
                queue_depth=self.output_q.qsize(),
                group_drops=self._group_drops,
            )
            if calibration_result is not None and calibration_result.phase in {"verify", "complete", "cached"}:
                self._status(
                    "NvFRUC calibration: "
                    f"phase={calibration_result.phase} "
                    f"safe={calibration_result.safe_output_fps} "
                    f"base={calibration_result.base_runtime_fps}"
                )
        generated_result = self._make_generated_result(
            current_result,
            generated_left,
            generated_right,
            (previous_ts + current_ts) * 0.5,
        )
        midpoint_ts = (previous_ts + current_ts) * 0.5
        self._put_group([
            (previous_result, previous_ts),
            (generated_result, midpoint_ts),
            (current_result, current_ts),
        ])
        self._real_frames += 2
        self._generated_frames += 1

    def run(self) -> None:
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = self.input_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                result, timestamp = item
                current = (result, self._timestamp(item))
                if self._previous is None:
                    self._previous = current
                    continue
                previous = self._previous
                self._previous = current
                try:
                    self._publish_group(previous, current)
                    now = time.perf_counter()
                    if now - self._last_report >= 5.0:
                        elapsed = max(1.0, now - self._last_report)
                        self._status(
                            "NvFRUC stats: "
                            f"base={self._real_frames / elapsed:.1f} "
                            f"generated={self._generated_frames / elapsed:.1f} "
                            f"queue={self.output_q.qsize()} group_drop={self._group_drops} "
                            f"resets={self._resets} "
                            f"process_ms={self._interpolate_seconds * 1000.0 / max(1, self._generated_frames):.1f}"
                        )
                        self._real_frames = 0
                        self._generated_frames = 0
                        self._group_drops = 0
                        self._resets = 0
                        self._interpolate_seconds = 0.0
                        self._last_report = now
                except NvFrucUnavailable as exc:
                    self._status(f"NvFRUC unavailable; passing through real frames: {exc}")
                    self._put_group([previous, current])
                    self._previous = None
                except Exception as exc:
                    self._status(
                        f"NvFRUC failed; resetting history and passing through real frames: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._resets += 1
                    if self._generator is not None:
                        try:
                            self._generator.reset()
                        except Exception:
                            pass
                    self._put_group([previous, current])
                    self._previous = None
        finally:
            self.close()

    def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._generator is not None:
            self._generator.close()
            self._generator = None
        self._previous = None
