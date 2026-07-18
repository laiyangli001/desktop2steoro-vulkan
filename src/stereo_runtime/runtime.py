from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from dataclasses import replace
import logging
import os
import time
from typing import Any

import torch

from .adapter import StereoRuntimeConfig, depth_provider_config_from_runtime, stereo_config_from_runtime
from .depth_postprocess import postprocess_depth
from .depth_provider import DepthProfileResult, create_depth_provider
from .openxr_render import OpenXRRenderConfig, render_openxr_stereo
from .parallax import parallax_debug_info, resolve_parallax_budget
from .render_size import runtime_output_size_text
from .settings_snapshot import (
    RuntimeSettingsPipelineRebuildRequired,
    RuntimeSettingsRestartRequired,
    RuntimeSettingsSnapshot,
    SnapshotChangeClass,
)
from .synthesis import StereoResult, synthesize_stereo
from .temporal import TemporalState


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthRuntimeResult:
    depth: torch.Tensor
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)


class DepthRuntime:
    """Persistent host-facing runtime for RGB frame -> depth only."""

    def __init__(
        self,
        config: StereoRuntimeConfig,
        *,
        depth_provider: Any | None = None,
        stats_window: int = 300,
        collect_memory_stats: bool = True,
    ) -> None:
        self.config = config
        self.depth_config = depth_provider_config_from_runtime(config)
        self.depth_provider = depth_provider if depth_provider is not None else create_depth_provider(self.depth_config)
        self._loaded = False
        self._active = True
        self.last_timing: dict[str, float] = {}
        self.last_memory: dict[str, float] = {}
        self.stats = RollingRuntimeStats(maxlen=stats_window)
        self.collect_memory_stats = bool(collect_memory_stats)

    def load(self) -> None:
        if self._loaded:
            return
        load = getattr(self.depth_provider, "load", None)
        if callable(load):
            load()
        self._loaded = True

    def set_inference_active(self, active: bool) -> None:
        self._active = bool(active)

    def reset_stats(self) -> None:
        self.stats.reset()
        self.last_timing = {}
        self.last_memory = {}

    def close(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self._loaded = False

    def provider_report(self) -> dict[str, Any]:
        return _provider_report(self.depth_provider)

    def to_report(self) -> dict[str, Any]:
        report = self.config.to_report()
        report["depth_provider"] = self.provider_report()
        report["depth_backend_resolved"] = self.depth_config.backend
        report["last_timing"] = dict(self.last_timing)
        report["last_memory"] = dict(self.last_memory)
        report["rolling_stats"] = self.stats.to_report()
        report["inference_active"] = self._active
        return report

    def predict_depth_frame(self, rgb_frame: torch.Tensor) -> DepthRuntimeResult:
        if not self._active:
            raise RuntimeError("DepthRuntime inference is paused")
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        total_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(total_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)
        return DepthRuntimeResult(depth=profile.depth, timing=timing, provider_info=self.provider_report())

    def _predict_depth_profile(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        predict_profile = getattr(self.depth_provider, "predict_profile", None)
        if callable(predict_profile):
            result = predict_profile(rgb_frame)
            if isinstance(result, DepthProfileResult):
                return result
            depth = getattr(result, "depth", None)
            if depth is not None:
                return DepthProfileResult(
                    depth=depth,
                    preprocess_ms=float(getattr(result, "preprocess_ms", 0.0)),
                    model_ms=float(getattr(result, "model_ms", 0.0)),
                    postprocess_ms=float(getattr(result, "postprocess_ms", 0.0)),
                    cuda_timing_events=dict(getattr(result, "cuda_timing_events", None) or {}),
                )

        start = time.perf_counter()
        depth = self.depth_provider.predict(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=float(elapsed), postprocess_ms=0.0)

    def _reset_cuda_peak_if_needed(self) -> None:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return
        device = self._runtime_cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _collect_memory_stats(self, rgb_frame: torch.Tensor) -> dict[str, float]:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return {}
        device = self._runtime_cuda_device(rgb_frame)
        if device is None:
            return {}
        try:
            return {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        except Exception:
            return {}

    def _runtime_cuda_device(self, rgb_frame: torch.Tensor | None = None) -> torch.device | None:
        if isinstance(rgb_frame, torch.Tensor) and rgb_frame.is_cuda:
            return rgb_frame.device
        try:
            device = torch.device(self.config.device)
        except Exception:
            return None
        return device if device.type == "cuda" else None


@dataclass(frozen=True)
class StereoRuntimeResult:
    depth: torch.Tensor
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    sbs: torch.Tensor
    output_eye_size: tuple[int, int] | None = None
    output_display_size: tuple[int, int] | None = None
    output_format: str | None = None
    output_dtype: str | None = None
    output_pack_backend: str | None = None
    active_settings_version: int | None = None
    hot_reload_class: str | None = None
    hot_reload_changed_fields: tuple[str, ...] = ()
    debug_info: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)
    cuda_ready_event: Any | None = None
    cuda_timing_events: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenXRRuntimeResult:
    depth: torch.Tensor
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    source_rgb: torch.Tensor | None = None
    output_eye_size: tuple[int, int] | None = None
    output_display_size: tuple[int, int] | None = None
    output_format: str | None = None
    output_dtype: str | None = None
    output_pack_backend: str | None = None
    active_settings_version: int | None = None
    hot_reload_class: str | None = None
    hot_reload_changed_fields: tuple[str, ...] = ()
    shader_uniforms: dict[str, Any] | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    provider_info: dict[str, Any] = field(default_factory=dict)
    cuda_ready_event: Any | None = None
    cuda_timing_events: dict[str, Any] = field(default_factory=dict)


def openxr_result_from_stereo_result(
    stereo_result: StereoRuntimeResult,
    source_rgb: torch.Tensor | None = None,
) -> OpenXRRuntimeResult:
    debug = dict(stereo_result.debug_info or {})
    left_eye = stereo_result.left_eye
    right_eye = stereo_result.right_eye
    cuda_events = dict(getattr(stereo_result, "cuda_timing_events", None) or {})
    display_size = _runtime_frame_size(left_eye)
    stereo_output_format = getattr(stereo_result, "output_format", None) or debug.get("runtime_output_format")
    if stereo_output_format == "half_sbs" and _should_split_half_sbs_for_openxr(debug):
        split_eyes = _split_half_sbs_frame(stereo_result.sbs)
        if split_eyes is not None:
            left_eye, right_eye = split_eyes
            display_size = _runtime_frame_size(stereo_result.sbs)
            debug.setdefault("runtime_output_pack_backend", "split_half_sbs")

    pack_start = time.perf_counter()
    _record_cuda_event(cuda_events, "openxr_pack_start", left_eye if isinstance(left_eye, torch.Tensor) else None)
    if _openxr_runtime_output_uint8_enabled():
        packed_left, left_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(left_eye)
        packed_right, right_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(right_eye)
        if packed_left is not left_eye or packed_right is not right_eye:
            left_eye = packed_left
            right_eye = packed_right
            pack_backend = _merge_openxr_rgba_pack_backend(left_pack_backend, right_pack_backend)
            previous = debug.get("runtime_output_pack_backend")
            debug["runtime_output_pack_backend"] = (
                pack_backend if previous in (None, "none") else f"{previous}+{pack_backend}"
            )
    _record_cuda_event(cuda_events, "openxr_pack", left_eye if isinstance(left_eye, torch.Tensor) else None)
    _record_cuda_event(cuda_events, "end", left_eye if isinstance(left_eye, torch.Tensor) else None)
    pack_ms = (time.perf_counter() - pack_start) * 1000.0

    debug["application_runtime_target"] = "openxr"
    debug["stereo_synthesis_mode"] = "full_synthesis_eyes"
    debug["runtime_output_format"] = "openxr_full_synthesis_eyes"
    debug["runtime_output_dtype"] = _runtime_eye_dtype(left_eye, right_eye)
    debug["runtime_output_eye_size"] = _runtime_eye_size(left_eye)
    debug["runtime_output_display_size"] = _runtime_size_text(display_size)
    debug.setdefault("runtime_output_pack_backend", "none")
    output_eye_size = _runtime_frame_size(left_eye)
    output_display_size = display_size
    timing = {**dict(stereo_result.timing or {}), "pack_ms": float(pack_ms)}

    return OpenXRRuntimeResult(
        depth=stereo_result.depth,
        left_eye=left_eye,
        right_eye=right_eye,
        source_rgb=source_rgb,
        output_eye_size=output_eye_size,
        output_display_size=output_display_size,
        output_format="openxr_full_synthesis_eyes",
        output_dtype=debug["runtime_output_dtype"],
        output_pack_backend=debug.get("runtime_output_pack_backend"),
        active_settings_version=getattr(stereo_result, "active_settings_version", None),
        hot_reload_class=getattr(stereo_result, "hot_reload_class", None),
        hot_reload_changed_fields=tuple(getattr(stereo_result, "hot_reload_changed_fields", ()) or ()),
        debug_info=debug,
        timing=timing,
        provider_info=dict(stereo_result.provider_info or {}),
        cuda_ready_event=getattr(stereo_result, "cuda_ready_event", None),
        cuda_timing_events=cuda_events,
    )



def _pack_openxr_eye_rgba_u8(eye: torch.Tensor) -> torch.Tensor:
    return _pack_openxr_eye_rgba_u8_with_backend(eye)[0]


def _pack_openxr_eye_rgba_u8_with_backend(eye: torch.Tensor) -> tuple[torch.Tensor, str]:
    if not isinstance(eye, torch.Tensor):
        return eye, "none"
    tensor = eye.detach()
    triton_packed = _try_pack_openxr_eye_rgba_u8_triton(tensor)
    if triton_packed is not None:
        return triton_packed, "triton_openxr_rgba_u8"
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            return eye, "none"
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
        tensor = tensor[:3].permute(1, 2, 0)
    elif tensor.ndim == 3 and tensor.shape[-1] == 4 and tensor.dtype == torch.uint8:
        return tensor.contiguous(), "torch_openxr_rgba_u8"
    elif tensor.ndim == 3 and tensor.shape[-1] >= 3:
        tensor = tensor[..., :3]
    else:
        return eye, "none"
    if tensor.is_floating_point():
        tensor = tensor.clamp(0.0, 1.0).mul(255.0).round()
    rgb = tensor.contiguous().clamp(0, 255).to(torch.uint8)
    h, w = rgb.shape[:2]
    rgba = torch.empty((h, w, 4), dtype=torch.uint8, device=rgb.device)
    rgba[..., :3].copy_(rgb[..., :3])
    rgba[..., 3].fill_(255)
    return rgba, "torch_openxr_rgba_u8"


def _try_pack_openxr_eye_rgba_u8_triton(tensor: torch.Tensor) -> torch.Tensor | None:
    if not (
        tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.ndim == 4
        and tensor.shape[0] == 1
        and tensor.shape[1] == 3
    ):
        return None
    try:
        from .output_triton import make_chw_rgb_to_hwc_rgba_u8

        return make_chw_rgb_to_hwc_rgba_u8(tensor)
    except Exception:
        return None


def _merge_openxr_rgba_pack_backend(left_backend: str, right_backend: str) -> str:
    if left_backend == right_backend:
        return left_backend
    backends = [backend for backend in (left_backend, right_backend) if backend and backend != "none"]
    return "+".join(backends) if backends else "none"


def _record_cuda_event(events: dict[str, Any], name: str, frame: torch.Tensor | None) -> None:
    if not isinstance(frame, torch.Tensor) or not frame.is_cuda:
        return
    try:
        event = torch.cuda.Event(blocking=False, enable_timing=True)
        event.record(torch.cuda.current_stream(frame.device))
        events[name] = event
    except Exception:
        return


def _should_split_half_sbs_for_openxr(debug: dict[str, Any]) -> bool:
    fused_backend = str(debug.get("fast_plus_fused_backend", "") or "").strip().lower()
    sbs_backend = str(debug.get("sbs_backend", "") or "").strip().lower()
    if fused_backend and fused_backend not in {"not_used", "none", "n/a"}:
        return True
    return "fused_half_sbs" in sbs_backend


def _split_half_sbs_frame(frame: Any) -> tuple[Any, Any] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) < 2:
        return None
    if len(shape) == 4 and shape[1] in (1, 3, 4):
        width_dim = 3
    elif len(shape) == 4 and shape[-1] in (1, 3, 4):
        width_dim = 2
    elif len(shape) == 3 and shape[0] in (1, 3, 4):
        width_dim = 2
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        width_dim = 1
    else:
        width_dim = len(shape) - 1
    width = int(shape[width_dim])
    half_width = width // 2
    if half_width <= 0:
        return None

    left_slice = [slice(None)] * len(shape)
    right_slice = [slice(None)] * len(shape)
    left_slice[width_dim] = slice(0, half_width)
    right_slice[width_dim] = slice(half_width, half_width * 2)
    left = frame[tuple(left_slice)]
    right = frame[tuple(right_slice)]
    if hasattr(left, "contiguous"):
        left = left.contiguous()
    if hasattr(right, "contiguous"):
        right = right.contiguous()
    return left, right


class RollingRuntimeStats:
    def __init__(self, *, maxlen: int = 300) -> None:
        self.maxlen = int(max(1, maxlen))
        self._samples: deque[dict[str, float]] = deque(maxlen=self.maxlen)
        self._memory_samples: deque[dict[str, float]] = deque(maxlen=self.maxlen)

    @property
    def count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self._memory_samples.clear()

    def update(self, timing: dict[str, float], memory: dict[str, float] | None = None) -> None:
        self._samples.append({key: float(value) for key, value in timing.items()})
        if memory:
            self._memory_samples.append({key: float(value) for key, value in memory.items()})

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "window": self.maxlen,
            "count": self.count,
            "stages": {},
            "fps": {},
            "memory": {},
        }
        if not self._samples:
            return report

        keys = sorted({key for sample in self._samples for key in sample})
        for key in keys:
            values = [sample[key] for sample in self._samples if key in sample]
            report["stages"][key] = _series_stats(values)

        total_values = [sample["total_ms"] for sample in self._samples if sample.get("total_ms", 0.0) > 0]
        if total_values:
            total_stats = _series_stats(total_values)
            report["fps"] = {
                "latest": 1000.0 / total_values[-1],
                "mean_from_mean_ms": 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0,
                "p90_from_p90_ms": 1000.0 / total_stats["p90"] if total_stats["p90"] > 0 else 0.0,
                "p99_from_p99_ms": 1000.0 / total_stats["p99"] if total_stats["p99"] > 0 else 0.0,
            }

        if self._memory_samples:
            memory_keys = sorted({key for sample in self._memory_samples for key in sample})
            for key in memory_keys:
                values = [sample[key] for sample in self._memory_samples if key in sample]
                report["memory"][key] = _series_stats(values)
        return report


def _series_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return {"count": 0.0, "latest": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "count": float(count),
        "latest": float(values[-1]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
        "median": _percentile_sorted(ordered, 0.50),
        "p90": _percentile_sorted(ordered, 0.90),
        "p99": _percentile_sorted(ordered, 0.99),
    }


def _percentile_sorted(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _snapshot_changed_fields(snapshot: RuntimeSettingsSnapshot) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in snapshot.__dataclass_fields__
            if getattr(snapshot, name) is not None and name not in {"version", "timestamp"}
        )
    )


def _merge_runtime_settings_snapshot(
    base: RuntimeSettingsSnapshot,
    updates: RuntimeSettingsSnapshot,
) -> RuntimeSettingsSnapshot:
    values = {"version": int(updates.version), "timestamp": float(updates.timestamp)}
    for name in updates.__dataclass_fields__:
        if name in {"version", "timestamp"}:
            continue
        value = getattr(updates, name)
        if value is not None:
            values[name] = value
        else:
            values[name] = getattr(base, name)
    return RuntimeSettingsSnapshot(**values)


_DEPTH_PROVIDER_REBUILD_FIELDS = frozenset(
    {"depth_backend", "model_id", "export_height", "export_width", "profile_sync", "use_cuda_graph"}
)
_RUNTIME_HANDLED_PIPELINE_REBUILD_FIELDS = _DEPTH_PROVIDER_REBUILD_FIELDS
_TEMPORAL_RESET_HOT_RELOAD_FIELDS = frozenset(
    {
        "temporal",
        "temporal_enabled",
        "depth_strength",
        "convergence",
        "max_disparity_px",
        "parallax_preset",
        "parallax_budget_preset",
        "foreground_shift_scale",
        "midground_shift_scale",
        "background_shift_scale",
        "dynamic_convergence_enabled",
        "dynamic_convergence_strength",
        "dynamic_convergence_target",
        "dynamic_convergence_alpha",
    }
)


def _append_temporal_reset_reason(debug: dict[str, Any], reason: str) -> None:
    current = debug.get("temporal_reset_reason")
    if not current:
        debug["temporal_reset_reason"] = reason
        return
    reasons = [part.strip() for part in str(current).split(",") if part.strip()]
    if reason not in reasons:
        reasons.append(reason)
    debug["temporal_reset_reason"] = ",".join(reasons)


def _consume_pending_temporal_reset_reasons(runtime: "StereoRuntime", debug: dict[str, Any]) -> None:
    for reason in runtime._pending_temporal_reset_reasons:
        _append_temporal_reset_reason(debug, reason)
    runtime._pending_temporal_reset_reasons = ()


def _add_active_settings_debug_info(debug: dict[str, Any], snapshot: RuntimeSettingsSnapshot) -> None:
    for field_name in (
        "source",
        "application_runtime_target",
        "runtime_quality_mode",
        "stereo_synthesis_mode",
        "render_size_policy",
        "stereo_render_scale",
        "output_transport",
        "presentation_flags",
        "debug_flags",
        "output_format",
        "max_disparity_px",
        "parallax_preset",
        "parallax_budget_preset",
        "convergence",
        "foreground_shift_scale",
        "midground_shift_scale",
        "background_shift_scale",
        "dynamic_convergence_enabled",
        "dynamic_convergence_strength",
        "dynamic_convergence_target",
        "dynamic_convergence_alpha",
        "hole_fill_mode",
    ):
        value = getattr(snapshot, field_name)
        if value is not None:
            debug[field_name] = value


def _add_runtime_config_debug_info(debug: dict[str, Any], config: StereoConfig) -> None:
    debug.setdefault("runtime_quality_mode", str(config.backend))
    debug.setdefault("output_format", str(config.output_format))
    debug.setdefault("stereo_synthesis_mode", "packed_synthesis")
    debug.setdefault("depth_strength", float(config.depth_strength))
    debug.setdefault("max_disparity_px", None if config.max_disparity_px is None else float(config.max_disparity_px))
    debug.setdefault("parallax_preset", str(config.parallax_preset))
    debug.setdefault("convergence", _debug_scalar_no_sync(config.convergence))
    debug.setdefault("foreground_shift_scale", float(getattr(config, "foreground_shift_scale", 1.0)))
    debug.setdefault("midground_shift_scale", float(getattr(config, "midground_shift_scale", 1.0)))
    debug.setdefault("background_shift_scale", float(getattr(config, "background_shift_scale", 1.0)))
    debug.setdefault("dynamic_convergence_enabled", bool(getattr(config, "dynamic_convergence_enabled", False)))


def _debug_scalar_no_sync(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return "cuda_tensor"
        return float(value.detach())
    return float(value)


def _add_depth_contract_debug_info(
    debug: dict[str, Any],
    depth: torch.Tensor,
    provider_info: dict[str, Any],
) -> None:
    debug["depth_render_size"] = runtime_output_size_text(_runtime_frame_size(depth))
    provider_size = _provider_size_label(provider_info)
    if provider_size is not None:
        debug["depth_provider_size"] = provider_size


def _layered_parallax_enabled(config: Any) -> bool:
    return any(
        abs(float(getattr(config, field_name, 1.0)) - 1.0) > 1e-6
        for field_name in ("foreground_shift_scale", "midground_shift_scale", "background_shift_scale")
    )


def _dynamic_convergence_config_for_depth(
    runtime: Any,
    depth: torch.Tensor,
    stereo_config: Any,
    *,
    prefer_gpu_tensor: bool = True,
) -> tuple[Any, dict[str, Any]]:
    enabled = bool(getattr(stereo_config, "dynamic_convergence_enabled", False))
    strength = max(0.0, min(1.0, float(getattr(stereo_config, "dynamic_convergence_strength", 0.0))))
    manual = float(getattr(stereo_config, "convergence", 0.0))
    if not enabled or strength <= 0.0:
        runtime._dynamic_convergence_value = None
        runtime._dynamic_convergence_last_measured = None
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
        return stereo_config, {"dynamic_convergence_effective": manual}
    target = max(0.0, min(1.0, float(getattr(stereo_config, "dynamic_convergence_target", 0.5))))
    alpha = max(0.0, min(0.98, float(getattr(stereo_config, "dynamic_convergence_alpha", 0.85))))
    measured = _dynamic_convergence_measurement(runtime, depth, target, prefer_gpu_tensor=prefer_gpu_tensor)
    previous = getattr(runtime, "_dynamic_convergence_value", None)
    if isinstance(measured, torch.Tensor):
        manual_tensor = measured.new_tensor(manual)
        desired = manual_tensor + (measured - manual_tensor) * strength
        if isinstance(previous, torch.Tensor) and previous.device == desired.device:
            effective = previous.to(dtype=desired.dtype) * alpha + desired * (1.0 - alpha)
        else:
            effective = desired
        effective = effective.detach()
        runtime._dynamic_convergence_value = effective
        runtime._dynamic_convergence_last_measured = measured.detach()
        return replace(stereo_config, convergence=effective), {
            "dynamic_convergence_effective": _debug_scalar_no_sync(effective),
            "dynamic_convergence_measured": _debug_scalar_no_sync(measured),
            "dynamic_convergence_manual": float(manual),
            "dynamic_convergence_strength": float(strength),
            "dynamic_convergence_target": float(target),
            "dynamic_convergence_alpha": float(alpha),
        }
    previous_float = _dynamic_convergence_previous_float(previous)
    if measured is None:
        effective = manual if previous_float is None else previous_float
    else:
        desired = manual + (measured - manual) * strength
        effective = desired if previous_float is None else previous_float * alpha + desired * (1.0 - alpha)
    runtime._dynamic_convergence_value = float(effective)
    runtime._dynamic_convergence_last_measured = measured
    return replace(stereo_config, convergence=float(effective)), {
        "dynamic_convergence_effective": float(effective),
        "dynamic_convergence_measured": None if measured is None else float(measured),
        "dynamic_convergence_manual": float(manual),
        "dynamic_convergence_strength": float(strength),
        "dynamic_convergence_target": float(target),
        "dynamic_convergence_alpha": float(alpha),
    }


def _dynamic_convergence_previous_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return None
        return float(value.detach())
    return float(value)


def _dynamic_convergence_measurement(
    runtime: Any,
    depth: torch.Tensor,
    quantile: float,
    *,
    prefer_gpu_tensor: bool = True,
) -> float | torch.Tensor | None:
    tensor = _depth_quantile_tensor(depth, quantile).detach().float()
    if not tensor.is_cuda:
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
        return float(tensor)
    if prefer_gpu_tensor:
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
        return tensor

    pending = getattr(runtime, "_dynamic_convergence_pending_measurement", None)
    event = getattr(runtime, "_dynamic_convergence_pending_event", None)
    if pending is not None and event is not None and event.query():
        runtime._dynamic_convergence_last_measured = float(pending.detach())
        runtime._dynamic_convergence_pending_measurement = None
        runtime._dynamic_convergence_pending_event = None
    if getattr(runtime, "_dynamic_convergence_pending_measurement", None) is None:
        pending = _cpu_scalar_buffer(tensor)
        pending.copy_(tensor, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(tensor.device))
        runtime._dynamic_convergence_pending_measurement = pending
        runtime._dynamic_convergence_pending_event = event
    return getattr(runtime, "_dynamic_convergence_last_measured", None)


def _cpu_scalar_buffer(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return torch.empty((), dtype=torch.float32, device="cpu", pin_memory=bool(getattr(tensor, "is_cuda", False)))
    except Exception:
        return torch.empty((), dtype=torch.float32, device="cpu")


def _depth_quantile_tensor(depth: torch.Tensor, quantile: float, *, max_samples: int = 8192) -> torch.Tensor:
    tensor = depth.detach().float().clamp(0.0, 1.0).flatten()
    if tensor.numel() == 0:
        return depth.new_tensor(0.0, dtype=torch.float32)
    if tensor.numel() > max_samples:
        stride = max(1, int(tensor.numel() // max_samples))
        tensor = tensor[::stride]
    count = int(tensor.numel())
    index = min(count - 1, max(0, int(round(float(quantile) * float(count - 1)))))
    return torch.sort(tensor).values[index]


def _provider_size_label(provider_info: dict[str, Any]) -> str | None:
    provider_size = provider_info.get("depth_provider_size")
    if provider_size is not None:
        return _size_label(provider_size, height_width=False)
    for key in ("input_size", "fixed_input_size"):
        value = provider_info.get(key)
        if value is not None:
            return _size_label(value, height_width=True)
    depth_resolution = provider_info.get("depth_resolution")
    if depth_resolution is None:
        return None
    try:
        size = int(depth_resolution)
    except (TypeError, ValueError):
        return str(depth_resolution)
    return f"{size}x{size}"


def _size_label(value, *, height_width: bool) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            first = int(value[0])
            second = int(value[1])
            return runtime_output_size_text((second, first) if height_width else (first, second))
        except (TypeError, ValueError):
            pass
    return str(value)


class StereoRuntime:
    """Persistent host-facing runtime for RGB frame -> depth -> stereo output."""

    def __init__(
        self,
        config: StereoRuntimeConfig,
        *,
        depth_provider: Any | None = None,
        temporal_state: TemporalState | None = None,
        stats_window: int = 300,
        collect_memory_stats: bool = True,
    ) -> None:
        self.config = config
        self.depth_config = depth_provider_config_from_runtime(config)
        self.stereo_config = stereo_config_from_runtime(config)
        self.depth_provider = depth_provider if depth_provider is not None else create_depth_provider(self.depth_config)
        self.temporal_state = temporal_state if temporal_state is not None else TemporalState()
        self._openxr_depth_temporal: torch.Tensor | None = None
        self._openxr_rgb_depth_dumped = False
        self._loaded = False
        self.last_timing: dict[str, float] = {}
        self.last_memory: dict[str, float] = {}
        self.active_settings_snapshot = RuntimeSettingsSnapshot(version=0, timestamp=0.0)
        self.active_settings_version = 0
        self.last_settings_change_class = SnapshotChangeClass.NO_CHANGE.value
        self.last_settings_changed_fields: tuple[str, ...] = ()
        self._pending_temporal_reset_reasons: tuple[str, ...] = ()
        self._last_runtime_perf_log_ts = 0.0
        self._runtime_frame_refresh_log_count = 0
        self._dynamic_convergence_value: float | None = None
        self._dynamic_convergence_last_measured: float | None = None
        self._dynamic_convergence_pending_measurement: torch.Tensor | None = None
        self._dynamic_convergence_pending_event: Any | None = None
        self.stats = RollingRuntimeStats(maxlen=stats_window)
        self.collect_memory_stats = bool(collect_memory_stats)

    def load(self) -> None:
        if self._loaded:
            return
        load = getattr(self.depth_provider, "load", None)
        if callable(load):
            load()
        self._loaded = True

    def reset_temporal(self) -> None:
        self.temporal_state.reset()

    def configure_stereo(self, stereo_config: Any, *, reset_temporal: bool = False) -> None:
        self.stereo_config = stereo_config
        if reset_temporal:
            self.temporal_state.reset_stereo()

    def apply_settings_snapshot(
        self,
        snapshot: RuntimeSettingsSnapshot,
        *,
        active_preset: str | None = None,
    ) -> SnapshotChangeClass:
        change_class = snapshot.classify()
        changed_fields = _snapshot_changed_fields(snapshot)
        merged_snapshot = _merge_runtime_settings_snapshot(self.active_settings_snapshot, snapshot)
        if change_class is SnapshotChangeClass.NO_CHANGE:
            self.active_settings_snapshot = merged_snapshot
            self.active_settings_version = int(snapshot.version)
            self.last_settings_change_class = change_class.value
            self.last_settings_changed_fields = changed_fields
            return change_class
        if change_class is SnapshotChangeClass.SESSION_RESTART:
            raise RuntimeSettingsRestartRequired(snapshot)
        if (
            change_class is SnapshotChangeClass.PIPELINE_REBUILD
            and not set(changed_fields).issubset(_RUNTIME_HANDLED_PIPELINE_REBUILD_FIELDS)
        ):
            raise RuntimeSettingsPipelineRebuildRequired(snapshot, changed_fields)

        updates = snapshot.to_config_updates()
        if active_preset is not None:
            updates["stereo_preset"] = active_preset
        self.config = replace(self.config, **updates)
        self.stereo_config = stereo_config_from_runtime(self.config)
        self.active_settings_snapshot = merged_snapshot
        self.active_settings_version = int(snapshot.version)
        self.last_settings_change_class = change_class.value
        self.last_settings_changed_fields = changed_fields

        if _TEMPORAL_RESET_HOT_RELOAD_FIELDS.intersection(changed_fields):
            self.temporal_state.reset_stereo()
            self._openxr_depth_temporal = None
            self._pending_temporal_reset_reasons = (*self._pending_temporal_reset_reasons, "settings_changed")

        if change_class is SnapshotChangeClass.PIPELINE_REBUILD and _DEPTH_PROVIDER_REBUILD_FIELDS.intersection(changed_fields):
            self._rebuild_depth_provider()
        return change_class

    def _rebuild_depth_provider(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self.depth_config = depth_provider_config_from_runtime(self.config)
        self.depth_provider = create_depth_provider(self.depth_config)
        self._loaded = False

    def warmup_stereo_kernels_for_frame(self, rgb_frame: torch.Tensor) -> None:
        """Compile stereo synthesis kernels for the actual runtime frame shape."""
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)
        device = rgb_frame.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        if str(os.environ.get("D2S_DISABLE_STEREO_WARMUP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            return
        self.load()
        preprocessor = getattr(self.depth_provider, "_preprocessor", None)
        engine = getattr(self.depth_provider, "_engine", None)
        if bool(getattr(self.config, "use_cuda_graph", False)) and preprocessor is not None and engine is not None:
            input_size = preprocessor.input_size(int(rgb_frame.shape[-2]), int(rgb_frame.shape[-1]))
            try:
                engine.capture_graph((1, 3, int(input_size[0]), int(input_size[1])))
            except RuntimeError as exc:
                setattr(self.depth_provider, "_cuda_graph_disabled_reason", f"{type(exc).__name__}: {exc}")
                clear_graph = getattr(engine, "clear_graph", None)
                if callable(clear_graph):
                    clear_graph()
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                print(
                    "\033[31m[TensorRT] CUDA graph warmup capture failed; disabled for this process. "
                    f"Using native TensorRT enqueue until restart: {type(exc).__name__}: {exc}\033[0m",
                    flush=True,
                )
                raise
        if rgb_frame.ndim == 3:
            _, height, width = rgb_frame.shape
        else:
            _, _, height, width = rgb_frame.shape
        rgb = torch.zeros((1, 3, int(height), int(width)), device=device, dtype=torch.float32)
        depth = torch.linspace(0.0, 1.0, int(width), device=device, dtype=torch.float32).view(1, 1, 1, int(width)).expand(1, 1, int(height), int(width)).contiguous()
        base = self.stereo_config
        depth_pop_values = {round(float(base.depth_pop), 3), 0.0, -0.7, 0.5}
        antialias_values = {round(float(base.depth_antialias_strength), 3), 0.0, 2.0}
        configs = []
        for depth_pop in sorted(depth_pop_values):
            for antialias in sorted(antialias_values):
                configs.append(
                    replace(
                        base,
                        temporal=False,
                        depth_pop=float(depth_pop),
                        depth_antialias_strength=float(antialias),
                    )
                )
        start = time.perf_counter()
        seen = set()
        for config in configs:
            key = (config.backend, config.output_format, config.layers, config.hole_fill, config.edge_dilation, round(float(config.depth_pop), 3), round(float(config.depth_antialias_strength), 3))
            if key in seen:
                continue
            seen.add(key)
            synthesize_stereo(rgb, depth, config, temporal_state=None)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[StereoRuntime] stereo kernel warmup complete: {len(seen)} configs {int(width)}x{int(height)} in {elapsed_ms:.1f}ms",
            flush=True,
        )
    def reset_stats(self) -> None:
        self.stats.reset()
        self.last_timing = {}
        self.last_memory = {}

    def close(self) -> None:
        close = getattr(self.depth_provider, "close", None)
        if callable(close):
            close()
        self._loaded = False

    def provider_report(self) -> dict[str, Any]:
        return _provider_report(self.depth_provider)

    def to_report(self) -> dict[str, Any]:
        report = self.config.to_report()
        report["depth_provider"] = self.provider_report()
        report["depth_backend_resolved"] = self.depth_config.backend
        report["stereo_backend"] = self.stereo_config.backend
        report["last_timing"] = dict(self.last_timing)
        report["last_memory"] = dict(self.last_memory)
        report["rolling_stats"] = self.stats.to_report()
        return report

    def process_rgb_frame(self, rgb_frame: torch.Tensor, *, skip_sbs_output: bool = False) -> StereoRuntimeResult:
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        cuda_events: dict[str, Any] = {}
        _record_cuda_event(cuda_events, "start", rgb_frame)
        total_start = time.perf_counter()
        depth_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        depth_total_ms = (time.perf_counter() - depth_start) * 1000.0
        depth = profile.depth
        cuda_events.update(getattr(profile, "cuda_timing_events", None) or {})
        _record_cuda_event(cuda_events, "depth", rgb_frame)
        stereo_config, convergence_debug = _dynamic_convergence_config_for_depth(self, depth, self.stereo_config)

        synth_start = time.perf_counter()
        fused_sbs, fused_skip = (None, "skip_sbs_output") if skip_sbs_output else self._try_fast_plus_fused_sbs(rgb_frame, depth, stereo_config)
        if fused_sbs is not None:
            stereo = StereoResult(
                left_eye=rgb_frame,
                right_eye=rgb_frame,
                sbs=fused_sbs,
                debug_info={
                    "backend": stereo_config.backend,
                    "sbs_backend": "triton_fast_plus_fused_half_sbs_uint8",
                    "fast_plus_fused_backend": "triton_half_sbs_uint8",
                    "fast_plus_fused_temporal_bypass": int(bool(stereo_config.temporal)),
                    "occlusion_mask_backend": "triton_fused_radius1",
                    "hole_fill_backend": "triton_fused_directional_4tap",
                },
            )
        else:
            stereo_config_for_frame = stereo_config
            if skip_sbs_output:
                stereo_config_for_frame = replace(stereo_config, output_format="mono")
            stereo = synthesize_stereo(
                rgb_frame,
                depth,
                stereo_config_for_frame,
                temporal_state=self.temporal_state,
            )
            cuda_events.update(getattr(stereo, "cuda_timing_events", None) or {})
            stereo.debug_info.setdefault("fast_plus_fused_backend", "not_used")
            stereo.debug_info.setdefault("fast_plus_fused_skip", fused_skip)
            if stereo_config_for_frame is not stereo_config:
                stereo.debug_info["sbs_backend"] = "openxr_eyes_only"
                stereo.debug_info["make_sbs_ms"] = 0.0
        synthesis_ms = (time.perf_counter() - synth_start) * 1000.0
        _record_cuda_event(cuda_events, "synthesis", rgb_frame)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(depth_total_ms),
            "synthesis_ms": float(synthesis_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)

        debug = dict(stereo.debug_info)
        debug["runtime_depth_backend"] = self.depth_config.backend
        debug["runtime_output_format"] = self.stereo_config.output_format
        debug["packing_format"] = self.stereo_config.output_format
        debug["runtime_depth_upsample"] = self.config.depth_upsample
        debug["active_settings_version"] = int(self.active_settings_version)
        debug["hot_reload_class"] = self.last_settings_change_class
        debug["hot_reload_changed_fields"] = list(self.last_settings_changed_fields)
        _consume_pending_temporal_reset_reasons(self, debug)
        _add_active_settings_debug_info(debug, self.active_settings_snapshot)
        _add_runtime_config_debug_info(debug, stereo_config)
        debug.update(convergence_debug)
        provider_info = self.provider_report()
        _add_depth_contract_debug_info(debug, depth, provider_info)
        _add_preprocess_debug_info(debug, rgb_frame)
        if memory:
            debug.update(memory)

        sbs = stereo.sbs
        pack_start = time.perf_counter()
        if skip_sbs_output:
            debug["runtime_output_pack_backend"] = "openxr_eyes_only"
            debug["runtime_output_dtype"] = str(sbs.dtype).replace("torch.", "")
        elif _runtime_output_uint8_enabled() and sbs.is_floating_point():
            fused_uint8 = _try_make_runtime_uint8_sbs(stereo, self.stereo_config.output_format)
            if fused_uint8 is not None:
                sbs = fused_uint8
                debug["runtime_output_pack_backend"] = "triton_half_sbs_uint8"
            else:
                sbs = sbs.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
                debug["runtime_output_pack_backend"] = "torch_float_to_uint8"
            debug["runtime_output_dtype"] = "uint8"
        else:
            if sbs.dtype == torch.uint8:
                debug.setdefault("runtime_output_pack_backend", debug.get("fast_plus_fused_backend", "prepacked_uint8"))
                debug["runtime_output_dtype"] = "uint8"
            else:
                debug["runtime_output_dtype"] = str(sbs.dtype).replace("torch.", "")
        output_eye_size, output_display_size = _add_runtime_output_size_debug_info(debug, stereo.left_eye, sbs)
        pack_ms = (time.perf_counter() - pack_start) * 1000.0
        _record_cuda_event(cuda_events, "pack", rgb_frame)
        _record_cuda_event(cuda_events, "end", rgb_frame)
        timing["pack_ms"] = float(pack_ms)
        slow_log_ms = float(os.environ.get("D2S_SLOW_RUNTIME_LOG_MS", "200") or "200")
        refresh_log_s = float(os.environ.get("D2S_RUNTIME_FRAME_LOG_REFRESH_S", "5") or "0")
        now_log = time.perf_counter()
        is_slow_frame = total_ms >= slow_log_ms
        is_refresh_frame = refresh_log_s > 0.0 and (now_log - self._last_runtime_perf_log_ts) >= refresh_log_s
        if is_slow_frame or is_refresh_frame:
            self._last_runtime_perf_log_ts = now_log
            depth_accounted_ms = float(profile.preprocess_ms) + float(profile.model_ms) + float(profile.postprocess_ms)
            depth_gap_ms = max(0.0, float(depth_total_ms) - depth_accounted_ms)
            log_kind = "slow frame" if is_slow_frame else "frame refresh"
            should_log_perf = True
            if log_kind == "frame refresh":
                should_log_perf = self._runtime_frame_refresh_log_count < 5
                if should_log_perf:
                    self._runtime_frame_refresh_log_count += 1
            if should_log_perf:
                LOGGER.debug(
                    f"[StereoRuntime] {log_kind}:"
                    f" total_ms={total_ms:.1f}"
                    f" depth_total_ms={depth_total_ms:.1f}"
                    f" depth_pre_ms={float(profile.preprocess_ms):.1f}"
                    f" depth_model_ms={float(profile.model_ms):.1f}"
                    f" depth_post_ms={float(profile.postprocess_ms):.1f}"
                    f" depth_gap_ms={depth_gap_ms:.1f}"
                    f" synthesis_ms={synthesis_ms:.1f}"
                    f" pack_ms={pack_ms:.1f}"
                    f" backend={debug.get('backend', stereo_config.backend)}"
                    f" depth_pop={stereo_config.depth_pop:.3f}"
                    f" antialias={stereo_config.depth_antialias_strength:.3f}"
                    f" output_dtype={debug.get('runtime_output_dtype', sbs.dtype)}"
                    f" pack_backend={debug.get('runtime_output_pack_backend', 'n/a')}"
                    f" sbs_backend={debug.get('sbs_backend', 'n/a')}"
                    f" fast_plus_fused={debug.get('fast_plus_fused_backend', 'n/a')}"
                    f" fast_plus_skip={debug.get('fast_plus_fused_skip', 'n/a')}"
                    f" stage_scene={float(debug.get('scene_detect_ms', 0.0)):.1f}"
                    f" stage_layered={float(debug.get('layered_total_ms', 0.0)):.1f}"
                    f" stage_depth_shift={float(debug.get('depth_postprocess_shift_ms', 0.0)):.1f}"
                    f" stage_warp={float(debug.get('warp_composite_ms', 0.0)):.1f}"
                    f" stage_occ={float(debug.get('occlusion_ms', 0.0)):.1f}"
                    f" stage_fill={float(debug.get('hole_fill_ms', 0.0)):.1f}"
                    f" stage_refine={float(debug.get('refine_ms', 0.0)):.1f}"
                    f" stage_temporal={float(debug.get('temporal_ms', 0.0)):.1f}"
                    f" stage_output_depth={float(debug.get('output_depth_ms', 0.0)):.1f}"
                    f" stage_sbs_backend={float(debug.get('sbs_backend_ms', 0.0)):.1f}"
                    f" stage_sbs={float(debug.get('make_sbs_ms', 0.0)):.1f}"
                    f" stage_synth_gap={float(debug.get('synthesis_unaccounted_ms', 0.0)):.1f}"
                )

        return StereoRuntimeResult(
            depth=depth,
            left_eye=stereo.left_eye,
            right_eye=stereo.right_eye,
            sbs=sbs,
            output_eye_size=output_eye_size,
            output_display_size=output_display_size,
            output_format=str(debug.get("runtime_output_format")),
            output_dtype=str(debug.get("runtime_output_dtype")),
            output_pack_backend=_optional_debug_str(debug.get("runtime_output_pack_backend")),
            active_settings_version=int(self.active_settings_version),
            hot_reload_class=self.last_settings_change_class,
            hot_reload_changed_fields=tuple(self.last_settings_changed_fields),
            debug_info=debug,
            timing=timing,
            provider_info=provider_info,
            cuda_ready_event=None,
            cuda_timing_events=cuda_events,
        )

    def process_openxr_frame(
        self,
        rgb_frame: torch.Tensor,
        openxr_config: OpenXRRenderConfig | None = None,
    ) -> OpenXRRuntimeResult:
        self.load()
        self._reset_cuda_peak_if_needed()
        rgb_frame = _validate_runtime_rgb_frame(rgb_frame)

        cuda_events: dict[str, Any] = {}
        _record_cuda_event(cuda_events, "start", rgb_frame)
        total_start = time.perf_counter()
        depth_start = time.perf_counter()
        profile = self._predict_depth_profile(rgb_frame)
        depth_total_ms = (time.perf_counter() - depth_start) * 1000.0
        depth = profile.depth
        output_mode = str(getattr(openxr_config, "output_mode", "auto") or "auto")
        if output_mode == "rgb_depth":
            prewarp_eyes = False
        elif output_mode == "full_synthesis_eyes":
            prewarp_eyes = True
        else:
            prewarp_eyes = _openxr_prewarp_eyes_enabled()
        stereo_config, convergence_debug = _dynamic_convergence_config_for_depth(
            self,
            depth,
            self.stereo_config,
            prefer_gpu_tensor=prewarp_eyes,
        )
        convergence = stereo_config.convergence
        if openxr_config is not None:
            openxr_config_for_frame = replace(
                openxr_config,
                convergence=convergence,
                foreground_shift_scale=float(getattr(stereo_config, "foreground_shift_scale", 1.0)),
                midground_shift_scale=float(getattr(stereo_config, "midground_shift_scale", 1.0)),
                background_shift_scale=float(getattr(stereo_config, "background_shift_scale", 1.0)),
            )
        else:
            openxr_config_for_frame = OpenXRRenderConfig(
                depth_strength=float(stereo_config.depth_strength),
                convergence=convergence,
                max_disparity_px=stereo_config.max_disparity_px,
                parallax_preset=str(stereo_config.parallax_preset),
                foreground_shift_scale=float(getattr(stereo_config, "foreground_shift_scale", 1.0)),
                midground_shift_scale=float(getattr(stereo_config, "midground_shift_scale", 1.0)),
                background_shift_scale=float(getattr(stereo_config, "background_shift_scale", 1.0)),
            )
        _record_cuda_event(cuda_events, "depth", rgb_frame)

        openxr_render_ms = 0.0
        pack_ms = 0.0
        pack_backend = "none"
        source_rgb = rgb_frame
        raw_depth = depth
        if prewarp_eyes:
            render_start = time.perf_counter()
            openxr = render_openxr_stereo(rgb_frame, depth, openxr_config_for_frame)
            openxr_render_ms = (time.perf_counter() - render_start) * 1000.0
            _record_cuda_event(cuda_events, "openxr_render", rgb_frame)

            pack_start = time.perf_counter()
            left_eye = openxr.left_eye
            right_eye = openxr.right_eye
            if bool(getattr(stereo_config, "cross_eyed", False)):
                left_eye, right_eye = right_eye, left_eye
            _record_cuda_event(cuda_events, "openxr_pack_start", left_eye)
            if _openxr_runtime_output_uint8_enabled():
                packed_left, left_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(left_eye)
                packed_right, right_pack_backend = _pack_openxr_eye_rgba_u8_with_backend(right_eye)
                if packed_left is not left_eye or packed_right is not right_eye:
                    left_eye = packed_left
                    right_eye = packed_right
                    pack_backend = _merge_openxr_rgba_pack_backend(left_pack_backend, right_pack_backend)
            pack_ms = (time.perf_counter() - pack_start) * 1000.0
            _record_cuda_event(cuda_events, "openxr_pack", left_eye)
            output_format = "openxr_eye_views"
            render_backend = dict(openxr.debug_info)
        else:
            depth = self._prepare_openxr_rgb_depth(depth)
            _record_cuda_event(cuda_events, "openxr_depth_prepare", rgb_frame)
            self._maybe_dump_openxr_rgb_depth(source_rgb=source_rgb, raw_depth=raw_depth, prepared_depth=depth)
            left_eye = rgb_frame
            right_eye = rgb_frame
            output_format = "openxr_rgb_depth"
            render_backend = {"backend": "openxr_viewer_shader_dibr"}
        total_ms = (time.perf_counter() - total_start) * 1000.0
        _record_cuda_event(cuda_events, "end", left_eye if isinstance(left_eye, torch.Tensor) else rgb_frame)

        timing = {
            "depth_preprocess_ms": float(profile.preprocess_ms),
            "depth_model_ms": float(profile.model_ms),
            "depth_postprocess_ms": float(profile.postprocess_ms),
            "depth_total_ms": float(depth_total_ms),
            "openxr_render_ms": float(openxr_render_ms),
            "pack_ms": float(pack_ms),
            "total_ms": float(total_ms),
        }
        memory = self._collect_memory_stats(rgb_frame)
        self.last_timing = timing
        self.last_memory = memory
        self.stats.update(timing, memory)

        debug = dict(render_backend)
        debug["application_runtime_target"] = "openxr"
        debug["stereo_synthesis_mode"] = "full_synthesis_eyes" if prewarp_eyes else "rgb_depth_direct"
        debug["runtime_depth_backend"] = self.depth_config.backend
        debug["runtime_output_format"] = output_format
        debug["packing_format"] = "none"
        debug["active_settings_version"] = int(self.active_settings_version)
        debug["runtime_output_dtype"] = _runtime_eye_dtype(left_eye, right_eye)
        debug["cross_eyed"] = int(bool(getattr(stereo_config, "cross_eyed", False)))
        debug["hot_reload_class"] = self.last_settings_change_class
        debug["hot_reload_changed_fields"] = list(self.last_settings_changed_fields)
        _consume_pending_temporal_reset_reasons(self, debug)
        _add_active_settings_debug_info(debug, self.active_settings_snapshot)
        _add_runtime_config_debug_info(debug, stereo_config)
        debug.update(convergence_debug)
        provider_info = self.provider_report()
        _add_depth_contract_debug_info(debug, depth, provider_info)
        output_eye_size, output_display_size = _add_runtime_output_size_debug_info(debug, left_eye, left_eye)
        debug["runtime_output_pack_backend"] = pack_backend
        shader_uniforms = None
        if openxr_config is not None:
            shader_uniforms = _add_openxr_config_debug_info(debug, openxr_config, left_eye)
        debug["runtime_depth_upsample"] = self.config.depth_upsample
        _add_preprocess_debug_info(debug, rgb_frame)
        if memory:
            debug.update(memory)
        if total_ms >= float(os.environ.get("D2S_SLOW_RUNTIME_LOG_MS", "120") or "120") and os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
            print(
                "[StereoRuntime] slow openxr frame:"
                f" total_ms={total_ms:.1f}"
                f" depth_total_ms={depth_total_ms:.1f}"
                f" depth_model_ms={float(profile.model_ms):.1f}"
                f" depth_postprocess_ms={float(profile.postprocess_ms):.1f}"
                f" openxr_render_ms={openxr_render_ms:.1f}"
                f" pack_ms={pack_ms:.1f}"
                f" output_dtype={debug.get('runtime_output_dtype', 'n/a')}"
                f" eye_size={debug.get('runtime_output_eye_size', 'n/a')}"
                f" render_backend={debug.get('backend', debug.get('openxr_backend', 'n/a'))}"
                f" depth_backend={debug.get('runtime_depth_backend', 'n/a')}",
                flush=True,
            )

        return OpenXRRuntimeResult(
            depth=depth,
            left_eye=left_eye,
            right_eye=right_eye,
            source_rgb=source_rgb,
            output_eye_size=output_eye_size,
            output_display_size=output_display_size,
            output_format=str(debug.get("runtime_output_format")),
            output_dtype=str(debug.get("runtime_output_dtype")),
            output_pack_backend=_optional_debug_str(debug.get("runtime_output_pack_backend")),
            active_settings_version=int(self.active_settings_version),
            hot_reload_class=self.last_settings_change_class,
            hot_reload_changed_fields=tuple(self.last_settings_changed_fields),
            shader_uniforms=shader_uniforms,
            debug_info=debug,
            timing=timing,
            provider_info=provider_info,
            cuda_timing_events=cuda_events,
        )

    def _prepare_openxr_rgb_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.detach().contiguous().float().clamp(0.0, 1.0)
        depth = _openxr_rgb_depth_percentile_normalize(depth, percentile=_openxr_rgb_depth_percentile())
        gamma = _openxr_rgb_depth_gamma()
        if abs(gamma - 1.0) > 1e-4:
            depth = depth.pow(gamma)
        depth = postprocess_depth(
            depth,
            depth_pop=float(getattr(self.stereo_config, "depth_pop", 0.0)),
            antialias_strength=float(getattr(self.stereo_config, "depth_antialias_strength", 0.0)),
        )
        return self._stabilize_openxr_rgb_depth(depth)

    def _maybe_dump_openxr_rgb_depth(
        self,
        *,
        source_rgb: torch.Tensor,
        raw_depth: torch.Tensor,
        prepared_depth: torch.Tensor,
    ) -> None:
        dump_dir = os.environ.get("D2S_OPENXR_RGB_DEPTH_DUMP_DIR", "").strip()
        if not dump_dir or self._openxr_rgb_depth_dumped:
            return
        try:
            from pathlib import Path

            from .io import save_depth, save_rgb

            out_dir = Path(dump_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_rgb(source_rgb.detach().float().clamp(0.0, 1.0), out_dir / "source_rgb.png")
            save_depth(raw_depth.detach().float().clamp(0.0, 1.0), out_dir / "raw_depth.png")
            save_depth(prepared_depth.detach().float().clamp(0.0, 1.0), out_dir / "prepared_depth.png")
            print(f"[StereoRuntime] OpenXR rgb_depth dump saved: {out_dir}", flush=True)
            self._openxr_rgb_depth_dumped = True
        except Exception as exc:
            print(f"[StereoRuntime] OpenXR rgb_depth dump failed: {type(exc).__name__}: {exc}", flush=True)

    def _stabilize_openxr_rgb_depth(self, depth: torch.Tensor) -> torch.Tensor:
        alpha = _openxr_rgb_depth_temporal_alpha()
        depth = depth.detach().contiguous().float()
        if alpha <= 0.0:
            self._openxr_depth_temporal = None
            return depth
        prev = self._openxr_depth_temporal
        if prev is None or prev.shape != depth.shape or prev.device != depth.device:
            self._openxr_depth_temporal = depth
            return depth
        out = prev.mul(alpha).add(depth, alpha=(1.0 - alpha))
        self._openxr_depth_temporal = out.detach()
        return out

    def _predict_depth_profile(self, rgb_frame: torch.Tensor) -> DepthProfileResult:
        predict_profile = getattr(self.depth_provider, "predict_profile", None)
        if callable(predict_profile):
            result = predict_profile(rgb_frame)
            if isinstance(result, DepthProfileResult):
                return result
            depth = getattr(result, "depth", None)
            if depth is not None:
                return DepthProfileResult(
                    depth=depth,
                    preprocess_ms=float(getattr(result, "preprocess_ms", 0.0)),
                    model_ms=float(getattr(result, "model_ms", 0.0)),
                    postprocess_ms=float(getattr(result, "postprocess_ms", 0.0)),
                )

        start = time.perf_counter()
        depth = self.depth_provider.predict(rgb_frame)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=float(elapsed), postprocess_ms=0.0)


    def _try_fast_plus_fused_sbs(self, rgb_frame: torch.Tensor, depth: torch.Tensor, stereo_config: Any) -> tuple[torch.Tensor | None, str]:
        if not _fast_plus_fused_enabled():
            return None, "disabled"
        if stereo_config.backend != "fast_plus":
            return None, f"backend={stereo_config.backend}"
        if stereo_config.output_format != "half_sbs":
            return None, f"format={stereo_config.output_format}"
        if not _runtime_output_uint8_enabled():
            return None, "runtime_uint8_off"
        if bool(getattr(stereo_config, "cross_eyed", False)):
            return None, "cross_eyed"
        if bool(getattr(stereo_config, "debug_output", False)):
            return None, "debug_output"
        if _layered_parallax_enabled(stereo_config):
            return None, "layered_parallax"
        if isinstance(getattr(stereo_config, "convergence", None), torch.Tensor):
            return None, "dynamic_convergence_tensor"
        try:
            from .fast_plus_fused_triton import can_use_fast_plus_fused_half_sbs_uint8, make_fast_plus_fused_half_sbs_uint8
            from .output import match_depth
        except Exception as exc:
            return None, f"import_failed:{type(exc).__name__}"
        depth = match_depth(depth, rgb_frame.shape[-2], rgb_frame.shape[-1])
        if not can_use_fast_plus_fused_half_sbs_uint8(rgb_frame, depth):
            return None, f"unsupported_tensor:rgb={tuple(rgb_frame.shape)}/{rgb_frame.dtype}/{rgb_frame.device};depth={tuple(depth.shape)}/{depth.dtype}/{depth.device}"
        budget = resolve_parallax_budget(
            render_width=int(rgb_frame.shape[-1]),
            render_height=int(rgb_frame.shape[-2]),
            preset=getattr(stereo_config, "parallax_preset", "standard"),
            convergence=float(getattr(stereo_config, "convergence", 0.0)),
            max_disparity_px=getattr(stereo_config, "max_disparity_px", None),
        )
        try:
            return make_fast_plus_fused_half_sbs_uint8(
                rgb_frame,
                depth,
                convergence=float(getattr(stereo_config, "convergence", 0.0)),
                max_disparity_px=float(budget.max_disparity_px),
                depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0))),
                edge_threshold=0.03,
            ), "used"
        except Exception as exc:
            return None, f"kernel_failed:{type(exc).__name__}"

    def _reset_cuda_peak_if_needed(self) -> None:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return
        device = self._runtime_cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    def _collect_memory_stats(self, rgb_frame: torch.Tensor) -> dict[str, float]:
        if not self.collect_memory_stats or not torch.cuda.is_available():
            return {}
        device = self._runtime_cuda_device(rgb_frame)
        if device is None:
            return {}
        try:
            return {
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                "cuda_peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
            }
        except Exception:
            return {}

    def _runtime_cuda_device(self, rgb_frame: torch.Tensor | None = None) -> torch.device | None:
        if isinstance(rgb_frame, torch.Tensor) and rgb_frame.is_cuda:
            return rgb_frame.device
        try:
            device = torch.device(self.config.device)
        except Exception:
            return None
        return device if device.type == "cuda" else None


StereoLabRuntime = StereoRuntime
StereoLabRuntimeResult = StereoRuntimeResult
StereoLabOpenXRRuntimeResult = OpenXRRuntimeResult
StereoLabDepthRuntime = DepthRuntime
StereoLabDepthRuntimeResult = DepthRuntimeResult


def _provider_report(depth_provider: Any) -> dict[str, Any]:
    info = getattr(depth_provider, "info", None)
    if info is None:
        return {}
    to_report = getattr(info, "to_report", None)
    if callable(to_report):
        return to_report()
    if isinstance(info, dict):
        return dict(info)
    return {"info": str(info)}


def _validate_runtime_rgb_frame(rgb_frame: Any) -> torch.Tensor:
    """Validate the capture/runtime boundary without doing capture adaptation."""
    if not isinstance(rgb_frame, torch.Tensor):
        raise TypeError("rgb_frame must be a torch.Tensor prepared by the capture layer")
    if rgb_frame.ndim not in (3, 4):
        raise ValueError(f"rgb_frame must be CHW or BCHW, got shape {tuple(rgb_frame.shape)}")
    channel_dim = 0 if rgb_frame.ndim == 3 else 1
    if rgb_frame.shape[channel_dim] != 3:
        raise ValueError(f"rgb_frame must be RGB with 3 channels in CHW/BCHW layout, got shape {tuple(rgb_frame.shape)}")
    if not rgb_frame.is_floating_point():
        raise TypeError(f"rgb_frame must be float 0..1; got dtype {rgb_frame.dtype}")
    return rgb_frame


def _add_runtime_output_size_debug_info(
    debug: dict[str, Any],
    eye_frame: torch.Tensor,
    display_frame: torch.Tensor,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    output_eye_size = _runtime_frame_size(eye_frame)
    output_display_size = _runtime_frame_size(display_frame)
    debug["runtime_output_eye_size"] = runtime_output_size_text(output_eye_size)
    debug["runtime_output_display_size"] = runtime_output_size_text(output_display_size)
    return output_eye_size, output_display_size


def _optional_debug_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _openxr_shader_uniforms(
    config: OpenXRRenderConfig,
    *,
    render_size: tuple[int, int] | None,
    max_disparity_px: float | None,
    depth_response: str | None,
) -> dict[str, Any]:
    return {
        "max_disparity_px": 0.0 if max_disparity_px is None else float(max_disparity_px),
        "parallax_preset": str(config.parallax_preset),
        "depth_response": str(depth_response or "unknown"),
        "depth_strength": float(config.depth_strength),
        "convergence": float(config.convergence),
        "foreground_shift_scale": float(getattr(config, "foreground_shift_scale", 1.0)),
        "midground_shift_scale": float(getattr(config, "midground_shift_scale", 1.0)),
        "background_shift_scale": float(getattr(config, "background_shift_scale", 1.0)),
        "render_size": None if render_size is None else (int(render_size[0]), int(render_size[1])),
        "screen_roll": float(config.screen_roll),
    }


def _add_openxr_config_debug_info(debug: dict[str, Any], config: OpenXRRenderConfig, eye_frame: torch.Tensor) -> dict[str, Any]:
    render_size = _runtime_frame_size(eye_frame)
    max_disparity_px = None
    depth_response = None
    if render_size is not None:
        budget = resolve_parallax_budget(
            render_width=render_size[0],
            render_height=render_size[1],
            preset=config.parallax_preset,
            convergence=config.convergence,
            max_disparity_px=config.max_disparity_px,
        )
        debug.update(parallax_debug_info(budget))
        max_disparity_px = float(budget.max_disparity_px)
        depth_response = str(budget.depth_response_name)
    elif config.max_disparity_px is not None:
        max_disparity_px = float(config.max_disparity_px)
        debug["resolved_max_disparity_px"] = max_disparity_px
        debug["parallax_budget_preset"] = str(config.parallax_preset)
    debug["openxr_convergence"] = float(config.convergence)
    if max_disparity_px is not None:
        debug["openxr_max_disparity_px"] = float(max_disparity_px)
    debug["openxr_parallax_preset"] = str(config.parallax_preset)
    debug["openxr_foreground_shift_scale"] = float(getattr(config, "foreground_shift_scale", 1.0))
    debug["openxr_midground_shift_scale"] = float(getattr(config, "midground_shift_scale", 1.0))
    debug["openxr_background_shift_scale"] = float(getattr(config, "background_shift_scale", 1.0))
    uniforms = _openxr_shader_uniforms(
        config,
        render_size=render_size,
        max_disparity_px=max_disparity_px,
        depth_response=depth_response or debug.get("depth_response"),
    )
    debug["openxr_shader_uniforms"] = uniforms
    return uniforms


def _add_preprocess_debug_info(debug: dict[str, Any], rgb_frame: torch.Tensor) -> None:
    mapping = {
        "_d2s_preprocess_backend": "preprocess_backend",
        "_d2s_preprocess_input_kind": "preprocess_input_kind",
        "_d2s_preprocess_device_origin": "preprocess_device_origin",
        "_d2s_preprocess_device_output": "preprocess_device_output",
        "_d2s_preprocess_device_transfer": "preprocess_device_transfer",
    }
    for attr, key in mapping.items():
        value = getattr(rgb_frame, attr, None)
        if value is not None:
            debug[key] = str(value)



def _runtime_frame_size(frame) -> tuple[int, int] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return int(shape[2]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return int(shape[1]), int(shape[0])
    if len(shape) >= 2:
        return int(shape[-1]), int(shape[-2])
    return None


def _runtime_size_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "unknown"
    return f"{int(size[0])}x{int(size[1])}"


def _runtime_eye_dtype(left_eye, right_eye) -> str:
    left_dtype = str(getattr(left_eye, "dtype", "unknown")).replace("torch.", "")
    right_dtype = str(getattr(right_eye, "dtype", "unknown")).replace("torch.", "")
    if left_dtype == right_dtype:
        return left_dtype
    return f"left={left_dtype},right={right_dtype}"


def _runtime_eye_size(eye) -> str:
    shape = tuple(getattr(eye, "shape", ()))
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3 and shape[-1] >= 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"
def _runtime_output_uint8_enabled() -> bool:
    return _env_flag("D2S_RUNTIME_OUTPUT_UINT8", "0")


def _openxr_runtime_output_uint8_enabled() -> bool:
    return _env_flag("D2S_OPENXR_RUNTIME_OUTPUT_UINT8", os.environ.get("D2S_RUNTIME_OUTPUT_UINT8", "1"))


def _openxr_prewarp_eyes_enabled() -> bool:
    return _env_flag("D2S_OPENXR_PREWARP_EYES", "0")


def _openxr_rgb_depth_temporal_alpha() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")
    try:
        return max(0.0, min(0.98, float(raw)))
    except Exception:
        return 0.9


def _openxr_rgb_depth_gamma() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_GAMMA", "1.2")
    try:
        return max(0.1, min(4.0, float(raw)))
    except Exception:
        return 1.2


def _openxr_rgb_depth_percentile() -> float:
    raw = os.environ.get("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "0")
    try:
        return max(0.0, min(20.0, float(raw)))
    except Exception:
        return 0.0


def _openxr_rgb_depth_percentile_normalize(depth: torch.Tensor, *, percentile: float) -> torch.Tensor:
    depth = depth.detach().contiguous().float().clamp(0.0, 1.0)
    if percentile <= 0.0:
        return depth
    flat = depth.flatten(start_dim=2)
    lo_q = max(0.0, min(1.0, float(percentile) / 100.0))
    hi_q = 1.0 - lo_q
    count = flat.shape[-1]
    if count <= 1:
        return depth
    lo_idx = min(count - 1, max(0, int(round(lo_q * (count - 1)))))
    hi_idx = min(count - 1, max(0, int(round(hi_q * (count - 1)))))
    sorted_vals = torch.sort(flat, dim=-1).values
    lo = sorted_vals[..., lo_idx].view(depth.shape[0], 1, 1, 1)
    hi = sorted_vals[..., hi_idx].view(depth.shape[0], 1, 1, 1)
    return ((depth - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _env_flag(name: str, default: object = "0") -> bool:
    return str(os.environ.get(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _fast_plus_fused_enabled() -> bool:
    return str(os.environ.get("D2S_FAST_PLUS_FUSED", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}


def _try_make_runtime_uint8_sbs(stereo: StereoResult, output_format: str) -> torch.Tensor | None:
    if output_format != "half_sbs":
        return None
    try:
        from .output_triton import can_use_triton_half_sbs, make_half_sbs_uint8
    except Exception:
        return None
    left = stereo.left_eye
    right = stereo.right_eye
    if not can_use_triton_half_sbs(left, right):
        return None
    try:
        return make_half_sbs_uint8(left, right)
    except Exception:
        return None
