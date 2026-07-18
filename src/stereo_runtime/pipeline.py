from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass, is_dataclass, replace
from typing import Callable

from capture.types import CapturedFrame

from .render_size import RenderSizeConfig, resolve_render_size, runtime_output_size_text
from .runtime import openxr_result_from_stereo_result
from .settings_snapshot import RuntimeSettingsPipelineRebuildRequired, RuntimeSettingsRestartRequired, RuntimeSettingsSnapshot

_OPENXR_FULL_SYNTHESIS_PRESETS = {"cinema", "game_low_latency", "still_image_hq", "debug_export"}


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_diag_stage() -> str:
    if _env_flag("D2S_RUNTIME_DROP_ONLY"):
        return "raw"
    return str(os.environ.get("D2S_RUNTIME_DIAG_STAGE", "full") or "full").strip().lower()


@dataclass(frozen=True)
class RuntimePipelineContext:
    shutdown_event: object
    raw_q: object
    runtime_q: object
    time_sleep: float
    run_mode: str
    openxr_runtime_direct: bool
    stereo_active_preset: str | None
    device: object
    use_cudart: bool
    thread_latencies: dict
    stereo_runtime: object
    capture_frame_to_rgb: Callable
    prepare_rgb_for_stereo_runtime: Callable
    current_openxr_render_config: Callable[[], object]
    is_hard_idle: Callable[[], bool]
    is_source_paused: Callable[[], bool]
    log_source_health: Callable[[], None]
    source_stat_inc: Callable[..., None]
    breakdown_inc: Callable[..., None]
    breakdown_add_time: Callable[..., None]
    breakdown_add_runtime_timing: Callable[..., None]
    set_preprocess_backend: Callable[[str], None]
    queue_clear: Callable[[object], None]
    queue_drain_latest: Callable[[object, object], object]
    queue_put_latest: Callable[[object, object], None]
    log_stereo_runtime_mode_once: Callable[[], None]
    apply_stereo_hot_reload_if_needed: Callable[[], None]
    warmup_stereo_once_for_frame: Callable[[object], None]
    log_fast_plus_fused_runtime_state: Callable[[object], None]
    application_runtime_target: str | None = None
    output_transport: str | None = None
    settings_update_q: object | None = None
    render_size_config: RenderSizeConfig | None = None


def _drain_latest_nowait(q: object | None):
    if q is None:
        return None
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


def _openxr_full_synthesis_enabled(ctx: RuntimePipelineContext) -> bool:
    if ctx.run_mode != "OpenXR":
        return False
    if not ctx.openxr_runtime_direct:
        return True
    return str(ctx.stereo_active_preset or "").strip().lower() in _OPENXR_FULL_SYNTHESIS_PRESETS


def _openxr_realtime_synthesis_config(config):
    if config is None:
        return None
    updates = {}
    if bool(getattr(config, "temporal", False)):
        updates.update(temporal=False, temporal_strength=0.0, auto_reset_temporal=False)
    return replace(config, **updates) if updates and is_dataclass(config) else config


def _active_preset_for_snapshot(settings_snapshot, active_preset):
    snapshot_preset = getattr(settings_snapshot, "stereo_preset", None)
    if snapshot_preset == "auto":
        return active_preset
    return snapshot_preset or active_preset


def _apply_latest_settings_snapshot(ctx: RuntimePipelineContext):
    settings_snapshot = _drain_latest_nowait(ctx.settings_update_q)
    if settings_snapshot is None:
        return None
    change_class = ctx.stereo_runtime.apply_settings_snapshot(
        settings_snapshot,
        active_preset=_active_preset_for_snapshot(settings_snapshot, ctx.stereo_active_preset),
    )
    ctx.source_stat_inc(
        "settings_updates",
        last_settings_version=int(settings_snapshot.version),
        last_settings_change_class=change_class.value,
    )
    return change_class


def _captured_frame_uses_cuda_capture(captured_frame: CapturedFrame | None) -> bool:
    return str(getattr(captured_frame, "capture_tool", "") or "").strip().lower() == "windowscapturecuda"


def _enable_openxr_depth_cuda_graph_if_needed(
    ctx: RuntimePipelineContext,
    openxr_full_synthesis: bool,
    captured_frame: CapturedFrame | None = None,
) -> None:
    if not openxr_full_synthesis:
        return
    if _captured_frame_uses_cuda_capture(captured_frame):
        ctx.source_stat_inc("openxr_depth_cuda_graph_skipped_cuda_capture")
        return
    config = getattr(ctx.stereo_runtime, "config", None)
    if config is None or bool(getattr(config, "use_cuda_graph", False)):
        return
    provider = getattr(ctx.stereo_runtime, "depth_provider", None)
    if getattr(provider, "_cuda_graph_disabled_reason", None):
        return
    backend = str(getattr(config, "depth_backend", "auto") or "auto").strip().lower()
    if backend not in {"auto", "tensorrt_native"}:
        return
    ctx.stereo_runtime.apply_settings_snapshot(
        RuntimeSettingsSnapshot(version=int(time.time() * 1000), timestamp=time.time(), use_cuda_graph=True),
        active_preset=ctx.stereo_active_preset,
    )
    ctx.source_stat_inc("openxr_depth_cuda_graph_enabled")


def _unpack_raw_queue_item(item):
    if isinstance(item, CapturedFrame):
        return item.frame, item.target_height, item.timestamp, item
    frame_raw, size, capture_start_time = item
    return frame_raw, size, capture_start_time, None


def _resolve_pipeline_render_size(size, config: RenderSizeConfig | None):
    if config is None:
        return size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return resolve_render_size((int(size[0]), int(size[1])), config)
    return size


def _reset_runtime_temporal_state(runtime) -> None:
    temporal_state = getattr(runtime, "temporal_state", None)
    reset_stereo = getattr(temporal_state, "reset_stereo", None)
    if callable(reset_stereo):
        reset_stereo()


def _source_target_key(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return ("legacy",)
    metadata = captured_frame.metadata if isinstance(captured_frame.metadata, dict) else {}
    metadata_parts = tuple(
        (key, str(metadata[key]))
        for key in ("source_id", "source_key", "capture_source", "capture_target", "target_id", "window_handle", "hwnd")
        if key in metadata and metadata[key] is not None
    )
    return (
        "captured",
        str(captured_frame.capture_mode or ""),
        int(captured_frame.monitor_index or 0),
        str(captured_frame.window_title or ""),
        metadata_parts,
    )


def _append_temporal_reset_reason(debug_info: dict, reason: str) -> None:
    current = debug_info.get("temporal_reset_reason")
    if not current:
        debug_info["temporal_reset_reason"] = reason
        return
    reasons = [part.strip() for part in str(current).split(",") if part.strip()]
    if reason not in reasons:
        reasons.append(reason)
    debug_info["temporal_reset_reason"] = ",".join(reasons)


def _rgb_size_text(frame) -> str:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        return f"{int(shape[3])}x{int(shape[2])}"
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"


def _capture_zero_copy(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return None
    if "zero_copy" in captured_frame.metadata:
        return bool(captured_frame.metadata["zero_copy"])
    return captured_frame.copy_mode.value == "none"


def _capture_copy_mode(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return None
    return captured_frame.copy_mode.value


def _capture_debug_fields(captured_frame: CapturedFrame | None, frame_rgb) -> dict:
    fields = {
        "preprocess_device_origin": getattr(frame_rgb, "_d2s_preprocess_device_origin", None),
        "preprocess_device_output": getattr(frame_rgb, "_d2s_preprocess_device_output", None),
        "preprocess_device_transfer": getattr(frame_rgb, "_d2s_preprocess_device_transfer", None),
        "preprocess_input_kind": getattr(frame_rgb, "_d2s_preprocess_input_kind", None),
        "capture_copy_mode": getattr(frame_rgb, "_d2s_capture_copy_mode", _capture_copy_mode(captured_frame)),
        "capture_zero_copy": getattr(frame_rgb, "_d2s_capture_zero_copy", _capture_zero_copy(captured_frame)),
    }
    if captured_frame is not None:
        fields.update(
            capture_tool=captured_frame.capture_tool,
            capture_frame_raw_device=captured_frame.frame_raw_device,
            capture_frame_raw_type=captured_frame.frame_raw_type,
            capture_frame_raw_dtype=captured_frame.frame_raw_dtype,
        )
    return {key: value for key, value in fields.items() if value is not None}


def _attach_capture_debug(runtime_result, captured_frame: CapturedFrame | None, frame_rgb) -> None:
    debug_info = getattr(runtime_result, "debug_info", None)
    if isinstance(debug_info, dict):
        debug_info.update(_capture_debug_fields(captured_frame, frame_rgb))


def _attach_pipeline_debug(
    runtime_result,
    *,
    capture_size,
    render_size,
    run_mode,
    render_size_config,
    application_runtime_target=None,
    output_transport=None,
) -> None:
    debug_info = getattr(runtime_result, "debug_info", None)
    if not isinstance(debug_info, dict):
        return
    debug_info["capture_size"] = _size_debug_text(capture_size)
    debug_info["render_size"] = _size_debug_text(render_size)
    debug_info["application_runtime_target"] = _application_target_debug_label(
        run_mode,
        application_runtime_target=application_runtime_target,
    )
    transport = _transport_debug_label(run_mode, output_transport=output_transport)
    debug_info["transport"] = transport
    debug_info["output_transport"] = transport
    if render_size_config is not None:
        debug_info["render_size_policy"] = render_size_config.policy.value
        debug_info["stereo_render_scale"] = render_size_config.scale_factor


def _application_target_debug_label(run_mode, *, application_runtime_target=None) -> str:
    if application_runtime_target:
        return str(application_runtime_target)
    if run_mode == "OpenXR":
        return "openxr"
    if run_mode in {"MJPEG", "RTMP", "Streamer", "Legacy Streamer", "MJPEG Streamer", "RTMP Streamer"}:
        return "network_stream"
    return "local_viewer"


def _transport_debug_label(run_mode, *, output_transport=None) -> str:
    if output_transport:
        return str(output_transport)
    return "openxr_swapchain" if run_mode == "OpenXR" else "local_window"


def _size_debug_text(size) -> str:
    if isinstance(size, (tuple, list)) and len(size) == 2:
        try:
            return runtime_output_size_text((int(size[0]), int(size[1])))
        except (TypeError, ValueError):
            pass
    return str(size)


def _run_depth_only(runtime, runtime_rgb) -> None:
    load = getattr(runtime, "load", None)
    if callable(load):
        load()
    predict = getattr(runtime, "_predict_depth_profile")
    predict(runtime_rgb)


def _synchronize_runtime_device(runtime_rgb) -> None:
    device = getattr(runtime_rgb, "device", None)
    if getattr(device, "type", None) != "cuda":
        return
    try:
        import torch

        torch.cuda.synchronize(device)
    except Exception:
        return


def _runtime_sync_after_frame_enabled(ctx: RuntimePipelineContext) -> bool:
    value = str(os.environ.get("D2S_RUNTIME_SYNC_AFTER_FRAME", "0") or "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", "auto"}:
        return False
    return False


def _runtime_result_cuda_device(runtime_result):
    for name in ("left_eye", "right_eye", "depth", "source_rgb"):
        tensor = getattr(runtime_result, name, None)
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) == "cuda":
            return device
    return None


def _attach_cuda_ready_event(runtime_result):
    if getattr(runtime_result, "cuda_ready_event", None) is not None:
        return runtime_result
    device = _runtime_result_cuda_device(runtime_result)
    if device is None:
        return runtime_result
    try:
        import torch

        event = torch.cuda.Event(blocking=False)
        event.record(torch.cuda.current_stream(device))
        if is_dataclass(runtime_result):
            return replace(runtime_result, cuda_ready_event=event)
        setattr(runtime_result, "cuda_ready_event", event)
    except Exception:
        return runtime_result
    return runtime_result


def _cuda_event_ready(event) -> bool:
    if event is None:
        return True
    query = getattr(event, "query", None)
    if not callable(query):
        return True
    try:
        return bool(query())
    except Exception:
        return True


def _runtime_pending_depth_limit() -> int:
    value = str(os.environ.get("D2S_RUNTIME_PENDING_CUDA_DEPTH", "1") or "1").strip()
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def _runtime_pending_cuda_wait_s(ctx) -> float:
    raw = os.environ.get("D2S_RUNTIME_PENDING_CUDA_WAIT_MS")
    if raw is None:
        raw = "0"
    try:
        return max(0.0, float(str(raw).strip()) / 1000.0)
    except ValueError:
        return 0.0


def _runtime_max_algorithm_fps(ctx: RuntimePipelineContext) -> float:
    default = "30" if ctx.run_mode == "OpenXR" else "0"
    raw = os.environ.get("D2S_RUNTIME_MAX_ALGO_FPS", default)
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 0.0


def _runtime_motion_gate_enabled(ctx: RuntimePipelineContext) -> bool:
    default = "0"
    return str(os.environ.get("D2S_RUNTIME_MOTION_GATE", default) or default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_motion_threshold(name: str, default: float) -> float:
    try:
        return max(0.0, float(str(os.environ.get(name, default)).strip()))
    except ValueError:
        return default


def _motion_sample(frame_rgb):
    shape = tuple(getattr(frame_rgb, "shape", ()))
    if len(shape) < 2:
        return None
    try:
        import torch
        if isinstance(frame_rgb, torch.Tensor):
            sample = frame_rgb.detach()
            if sample.ndim == 4:
                sample = sample[0]
            if sample.ndim == 3 and sample.shape[0] in (3, 4):
                h, w = int(sample.shape[1]), int(sample.shape[2])
                sample = sample[:, ::max(1, h // 18), ::max(1, w // 32)]
            elif sample.ndim >= 2:
                h, w = int(sample.shape[0]), int(sample.shape[1])
                sample = sample[::max(1, h // 18), ::max(1, w // 32)]
            is_integer = not torch.is_floating_point(sample)
            sample = sample.to(dtype=torch.float32)
            return sample / 255.0 if is_integer else sample
    except Exception:
        pass
    try:
        import numpy as np
        sample = np.asarray(frame_rgb)
        if sample.ndim == 4:
            sample = sample[0]
        if sample.ndim == 3 and sample.shape[0] in (3, 4):
            h, w = int(sample.shape[1]), int(sample.shape[2])
            sample = sample[:, ::max(1, h // 18), ::max(1, w // 32)]
        elif sample.ndim >= 2:
            h, w = int(sample.shape[0]), int(sample.shape[1])
            sample = sample[::max(1, h // 18), ::max(1, w // 32)]
        is_integer = np.issubdtype(sample.dtype, np.integer)
        sample = sample.astype("float32", copy=False)
        return sample / 255.0 if is_integer else sample
    except Exception:
        return None


def _motion_score(previous, current) -> float | None:
    if previous is None or current is None:
        return None
    if tuple(getattr(previous, "shape", ())) != tuple(getattr(current, "shape", ())):
        return None
    try:
        import torch
        if isinstance(current, torch.Tensor):
            return float((current - previous).abs().mean().item())
    except Exception:
        pass
    try:
        import numpy as np
        return float(np.mean(np.abs(current - previous)))
    except Exception:
        return None


def _is_fatal_runtime_preparation_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    text = str(exc or "").lower()
    if not text:
        return False
    fatal_markers = (
        "unable to resolve infinidepth weights",
        "model directory not found",
        "onnx artifact not found",
        "tensorrt engine not found",
        "download completed but model directory was not found",
        "model weights resolved but model directory was not found",
    )
    return any(marker in text for marker in fatal_markers)


def _cuda_elapsed_ms(events: dict, start: str, end: str) -> float | None:
    first = events.get(start)
    second = events.get(end)
    if first is None or second is None:
        return None
    elapsed_time = getattr(first, "elapsed_time", None)
    if not callable(elapsed_time):
        return None
    try:
        return float(elapsed_time(second))
    except Exception:
        return None


def _add_cuda_event_timings(ctx: RuntimePipelineContext, runtime_result) -> None:
    events = getattr(runtime_result, "cuda_timing_events", None)
    if not isinstance(events, dict) or not events:
        return
    for name, start, end in (
        ("rt_gpu_depth", "start", "depth"),
        ("rt_gpu_depth_preprocess", "depth_pre_start", "depth_pre_end"),
        ("rt_gpu_depth_model", "depth_model_start", "depth_model_end"),
        ("rt_gpu_depth_normalize", "depth_norm_start", "depth_norm_end"),
        ("rt_gpu_depth_upsample", "depth_upsample_start", "depth_upsample_end"),
        ("rt_gpu_depth_postprocess", "depth_post_start", "depth_post_end"),
        ("rt_gpu_synth", "depth", "synthesis"),
        ("rt_gpu_synth_scene", "synth_start", "synth_scene"),
        ("rt_gpu_synth_depth_shift", "synth_scene", "synth_depth_shift"),
        ("rt_gpu_synth_warp", "synth_depth_shift", "synth_warp"),
        ("rt_gpu_synth_occ", "synth_warp", "synth_occlusion"),
        ("rt_gpu_synth_fill", "synth_occlusion", "synth_hole_fill"),
        ("rt_gpu_synth_refine", "synth_hole_fill", "synth_refine"),
        ("rt_gpu_synth_temporal", "synth_refine", "synth_temporal"),
        ("rt_gpu_synth_output_depth", "synth_temporal", "synth_output_depth"),
        ("rt_gpu_synth_sbs", "synth_output_depth", "synth_sbs"),
        ("rt_gpu_pack", "synthesis", "pack"),
        ("rt_gpu_openxr_pack", "openxr_pack_start", "openxr_pack"),
        ("rt_gpu_total", "start", "end"),
    ):
        elapsed_ms = _cuda_elapsed_ms(events, start, end)
        if elapsed_ms is not None:
            ctx.breakdown_add_time(name, elapsed_ms / 1000.0)


class RuntimePipelineLoop:
    def __init__(self, context: RuntimePipelineContext):
        self.context = context
        self._logged_rgb_shape = False
        self._logged_drop_only = False
        self._last_render_size = None
        self._last_source_target_key = None
        self._has_source_target_key = False
        self._last_cuda_ready_event = None
        self._pending_runtime_items = []
        self._last_runtime_motion_sample = None
        self._last_algorithm_output_time = 0.0
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        load = getattr(self.context.stereo_runtime, "load", None)
        if callable(load):
            load()
        self._prepared = True

    def _publish_runtime_item(self, item) -> None:
        ctx = self.context
        runtime_result, capture_start_time, process_latency, runtime_latency, pending_since = item
        queue_put_start_time = time.perf_counter()
        _add_cuda_event_timings(ctx, runtime_result)
        try:
            runtime_q_was_full = ctx.runtime_q.full()
        except Exception:
            runtime_q_was_full = False
        ctx.queue_put_latest(ctx.runtime_q, (runtime_result, capture_start_time))
        if runtime_q_was_full:
            ctx.breakdown_inc("runtime_overwrite")
            ctx.source_stat_inc("runtime_overwrite")
        ctx.breakdown_add_time("rt_put", time.perf_counter() - queue_put_start_time)
        if pending_since is not None:
            ctx.breakdown_add_time("rt_pending_age", time.perf_counter() - pending_since)
        self._last_algorithm_output_time = time.perf_counter()
        ctx.source_stat_inc(
            "runtime_frames",
            last_runtime_ts=time.perf_counter(),
            last_process_latency=process_latency,
            last_runtime_latency=runtime_latency,
        )
        ctx.breakdown_inc("runtime")

    def _publish_ready_pending_items(self) -> int:
        if not self._pending_runtime_items:
            return 0
        ready_index = None
        for index in range(len(self._pending_runtime_items) - 1, -1, -1):
            pending_result = self._pending_runtime_items[index][0]
            if _cuda_event_ready(getattr(pending_result, "cuda_ready_event", None)):
                ready_index = index
                break
        if ready_index is None:
            pending_limit = _runtime_pending_depth_limit()
            if len(self._pending_runtime_items) > pending_limit:
                self._pending_runtime_items[:] = self._pending_runtime_items[-pending_limit:]
            return 0
        item = self._pending_runtime_items[ready_index]
        self._pending_runtime_items[:] = self._pending_runtime_items[ready_index + 1:]
        self._publish_runtime_item(item)
        return 1

    def _publish_ready_pending_items_until(self, timeout_s: float) -> int:
        published = self._publish_ready_pending_items()
        if published:
            ctx = self.context
            ctx.source_stat_inc("runtime_pending_cuda_wait")
            ctx.breakdown_inc("runtime_pending_cuda_wait")
            ctx.breakdown_add_time("rt_pending_wait", 0.0)
            return published
        if timeout_s <= 0.0 or not self._pending_runtime_items:
            return published
        ctx = self.context
        wait_start = time.perf_counter()
        deadline = wait_start + timeout_s
        while self._pending_runtime_items and not ctx.shutdown_event.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            time.sleep(min(max(remaining, 0.0), max(float(ctx.time_sleep), 0.0005), 0.0005))
            published = self._publish_ready_pending_items()
            if published:
                ctx.source_stat_inc("runtime_pending_cuda_wait")
                ctx.breakdown_inc("runtime_pending_cuda_wait")
                ctx.breakdown_add_time("rt_pending_wait", time.perf_counter() - wait_start)
                return published
        ctx.breakdown_add_time("rt_pending_wait", time.perf_counter() - wait_start)
        return 0

    def run(self) -> None:
        ctx = self.context
        while not ctx.shutdown_event.is_set():
            ctx.log_source_health()
            try:
                diag_stage = _runtime_diag_stage()
                if ctx.shutdown_event.is_set():
                    break
                if ctx.is_hard_idle():
                    self._pending_runtime_items.clear()
                    self._last_cuda_ready_event = None
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    time.sleep(0.1)
                    continue
                if ctx.is_source_paused():
                    self._pending_runtime_items.clear()
                    self._last_cuda_ready_event = None
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue
                self._publish_ready_pending_items()
                if len(self._pending_runtime_items) >= _runtime_pending_depth_limit():
                    if self._publish_ready_pending_items_until(_runtime_pending_cuda_wait_s(ctx)):
                        continue
                    try:
                        ctx.queue_drain_latest(ctx.raw_q, ctx.raw_q.get_nowait())
                        ctx.source_stat_inc("raw_get", last_raw_get_ts=time.perf_counter())
                        ctx.breakdown_inc("raw_get")
                        ctx.source_stat_inc("runtime_drop_cuda_inflight")
                        ctx.breakdown_inc("runtime_drop_cuda_inflight")
                    except queue.Empty:
                        ctx.source_stat_inc("runtime_pending_cuda_inflight")
                        ctx.breakdown_inc("runtime_pending_cuda_inflight")
                        time.sleep(min(ctx.time_sleep, 0.001))
                    continue

                _apply_latest_settings_snapshot(ctx)

                frame_raw, size, capture_start_time, captured_frame = _unpack_raw_queue_item(
                    ctx.queue_drain_latest(
                        ctx.raw_q,
                        ctx.raw_q.get(timeout=min(ctx.time_sleep, 0.01)),
                    )
                )
                ctx.source_stat_inc("raw_get", last_raw_get_ts=time.perf_counter())
                ctx.breakdown_inc("raw_get")

                if diag_stage == "raw":
                    if not self._logged_drop_only:
                        self._logged_drop_only = True
                        print("[RuntimePipeline] diag_stage=raw: dropping raw frames before GPU work", flush=True)
                    ctx.source_stat_inc("runtime_diag_raw")
                    ctx.breakdown_inc("runtime_diag_raw")
                    continue

                if len(self._pending_runtime_items) >= _runtime_pending_depth_limit() and not _cuda_event_ready(self._last_cuda_ready_event):
                    ctx.source_stat_inc("runtime_drop_cuda_inflight")
                    ctx.breakdown_inc("runtime_drop_cuda_inflight")
                    continue
                self._last_cuda_ready_event = None

                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                loop_start_time = time.perf_counter()

                process_start_time = time.perf_counter()
                render_size = _resolve_pipeline_render_size(size, ctx.render_size_config)
                render_size_changed = self._last_render_size is not None and render_size != self._last_render_size
                source_target_key = _source_target_key(captured_frame)
                source_target_changed = (
                    self._has_source_target_key
                    and source_target_key != self._last_source_target_key
                )
                if render_size_changed:
                    _reset_runtime_temporal_state(ctx.stereo_runtime)
                if source_target_changed:
                    _reset_runtime_temporal_state(ctx.stereo_runtime)
                self._last_render_size = render_size
                self._last_source_target_key = source_target_key
                self._has_source_target_key = True
                frame_rgb = ctx.capture_frame_to_rgb(
                    frame_raw,
                    render_size,
                    device=ctx.device,
                    use_torch=ctx.use_cudart,
                    output="tensor",
                    frame_raw_device=captured_frame.frame_raw_device if captured_frame else None,
                    capture_copy_mode=_capture_copy_mode(captured_frame),
                    capture_zero_copy=_capture_zero_copy(captured_frame),
                )
                if not self._logged_rgb_shape and os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                    self._logged_rgb_shape = True
                    print(
                        f"[process_runtime_loop] rgb={_rgb_size_text(frame_rgb)}",
                        flush=True,
                    )
                ctx.breakdown_add_time("rt_cap2rgb", time.perf_counter() - process_start_time)
                ctx.set_preprocess_backend(
                    str(getattr(frame_rgb, "_d2s_preprocess_backend", "unknown"))
                )
                if diag_stage == "preprocess":
                    ctx.source_stat_inc("runtime_diag_preprocess")
                    ctx.breakdown_inc("runtime_diag_preprocess")
                    continue
                process_latency = process_start_time - capture_start_time
                ctx.thread_latencies["capture"] = process_latency

                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                runtime_start_time = time.perf_counter()
                prepare_start_time = time.perf_counter()
                runtime_rgb = ctx.prepare_rgb_for_stereo_runtime(frame_rgb, device=ctx.device)
                ctx.breakdown_add_time("rt_prepare", time.perf_counter() - prepare_start_time)
                if ctx.run_mode == "OpenXR" and _runtime_motion_gate_enabled(ctx):
                    motion_sample = _motion_sample(runtime_rgb)
                    motion = _motion_score(self._last_runtime_motion_sample, motion_sample)
                    now_for_gate = time.perf_counter()
                    max_fps = _runtime_max_algorithm_fps(ctx)
                    min_interval = 1.0 / max_fps if max_fps > 0.0 else 0.0
                    small_threshold = _runtime_motion_threshold("D2S_RUNTIME_MOTION_SMALL_THRESHOLD", 0.003)
                    force_threshold = _runtime_motion_threshold("D2S_RUNTIME_MOTION_FORCE_THRESHOLD", 0.02)
                    too_soon = min_interval > 0.0 and (now_for_gate - self._last_algorithm_output_time) < min_interval
                    weak_motion = motion is not None and motion < small_threshold
                    strong_motion = motion is not None and motion >= force_threshold
                    if self._last_runtime_motion_sample is not None and (weak_motion or (too_soon and not strong_motion)):
                        ctx.source_stat_inc("runtime_motion_skip")
                        ctx.breakdown_inc("runtime_motion_skip")
                        continue
                    self._last_runtime_motion_sample = motion_sample
                if diag_stage == "prepare":
                    ctx.source_stat_inc("runtime_diag_prepare")
                    ctx.breakdown_inc("runtime_diag_prepare")
                    continue
                if diag_stage == "depth":
                    depth_start_time = time.perf_counter()
                    _run_depth_only(ctx.stereo_runtime, runtime_rgb)
                    ctx.breakdown_add_time("rt_call", time.perf_counter() - depth_start_time)
                    ctx.source_stat_inc("runtime_diag_depth")
                    ctx.breakdown_inc("runtime_diag_depth")
                    continue
                ctx.log_stereo_runtime_mode_once()
                ctx.apply_stereo_hot_reload_if_needed()
                _apply_latest_settings_snapshot(ctx)
                openxr_full_synthesis = ctx.run_mode == "OpenXR" and _openxr_full_synthesis_enabled(ctx)
                _enable_openxr_depth_cuda_graph_if_needed(ctx, openxr_full_synthesis, captured_frame)
                runtime_call_start_time = time.perf_counter()
                if ctx.run_mode == "OpenXR" and not openxr_full_synthesis:
                    runtime_result = ctx.stereo_runtime.process_openxr_frame(
                        runtime_rgb,
                        ctx.current_openxr_render_config(),
                    )
                else:
                    original_stereo_config = None
                    if openxr_full_synthesis:
                        original_stereo_config = getattr(ctx.stereo_runtime, "stereo_config", None)
                        realtime_config = _openxr_realtime_synthesis_config(original_stereo_config)
                        if realtime_config is not original_stereo_config:
                            ctx.stereo_runtime.stereo_config = realtime_config
                    try:
                        runtime_result = ctx.stereo_runtime.process_rgb_frame(
                            runtime_rgb,
                            skip_sbs_output=openxr_full_synthesis,
                        )
                    finally:
                        if original_stereo_config is not None:
                            ctx.stereo_runtime.stereo_config = original_stereo_config
                    if ctx.run_mode == "OpenXR":
                        runtime_result = openxr_result_from_stereo_result(runtime_result, source_rgb=runtime_rgb)
                ctx.breakdown_add_time("rt_call", time.perf_counter() - runtime_call_start_time)
                _attach_pipeline_debug(
                    runtime_result,
                    capture_size=size,
                    render_size=render_size,
                    run_mode=ctx.run_mode,
                    render_size_config=ctx.render_size_config,
                    application_runtime_target=ctx.application_runtime_target,
                    output_transport=ctx.output_transport,
                )
                debug_info = getattr(runtime_result, "debug_info", None)
                if isinstance(debug_info, dict):
                    if render_size_changed:
                        _append_temporal_reset_reason(debug_info, "render_size_changed")
                    if source_target_changed:
                        _append_temporal_reset_reason(debug_info, "source_target_changed")
                    if ctx.application_runtime_target:
                        debug_info["application_runtime_target"] = ctx.application_runtime_target
                    if ctx.output_transport:
                        debug_info["output_transport"] = ctx.output_transport
                _attach_capture_debug(runtime_result, captured_frame, frame_rgb)
                ctx.breakdown_add_runtime_timing(runtime_result)
                ctx.log_fast_plus_fused_runtime_state(runtime_result)
                if runtime_result.depth is None:
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_none")
                    continue
                ctx.warmup_stereo_once_for_frame(runtime_rgb)
                if diag_stage == "runtime_sync" or _runtime_sync_after_frame_enabled(ctx):
                    sync_start_time = time.perf_counter()
                    _synchronize_runtime_device(runtime_rgb)
                    ctx.breakdown_add_time("rt_sync", time.perf_counter() - sync_start_time)
                    ctx.source_stat_inc("runtime_gpu_sync")
                if diag_stage in {"runtime", "runtime_sync"}:
                    ctx.source_stat_inc("runtime_diag_runtime")
                    ctx.breakdown_inc("runtime_diag_runtime")
                    continue
                runtime_result = _attach_cuda_ready_event(runtime_result)
                self._last_cuda_ready_event = getattr(runtime_result, "cuda_ready_event", None)
                runtime_latency = time.perf_counter() - runtime_start_time
                ctx.thread_latencies["resize"] = process_latency
                ctx.thread_latencies["runtime"] = runtime_latency

                if not _cuda_event_ready(self._last_cuda_ready_event):
                    self._pending_runtime_items.append(
                        (runtime_result, capture_start_time, process_latency, runtime_latency, time.perf_counter())
                    )
                    pending_limit = _runtime_pending_depth_limit()
                    if len(self._pending_runtime_items) > pending_limit:
                        self._pending_runtime_items[:] = self._pending_runtime_items[-pending_limit:]
                    ctx.breakdown_inc("runtime_pending_cuda")
                    ctx.source_stat_inc("runtime_pending_cuda")
                    ctx.breakdown_add_time("rt_loop", time.perf_counter() - loop_start_time)
                    continue

                self._publish_runtime_item(
                    (runtime_result, capture_start_time, process_latency, runtime_latency, None)
                )
                ctx.breakdown_add_time("rt_loop", time.perf_counter() - loop_start_time)

            except queue.Empty:
                ctx.source_stat_inc("raw_queue_empty")
                continue
            except (RuntimeSettingsPipelineRebuildRequired, RuntimeSettingsRestartRequired):
                raise
            except Exception as exc:
                fatal_error = _is_fatal_runtime_preparation_error(exc)
                ctx.source_stat_inc(
                    "runtime_errors",
                    last_error=f"process_runtime_loop {type(exc).__name__}: {exc}",
                )
                if fatal_error:
                    ctx.source_stat_inc("runtime_fatal_errors")
                    print(f"[process_runtime_loop] Fatal: {type(exc).__name__}: {exc}", flush=True)
                    try:
                        ctx.shutdown_event.set()
                    except Exception:
                        pass
                    break
                print(f"[process_runtime_loop] Error: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.05)
                continue
