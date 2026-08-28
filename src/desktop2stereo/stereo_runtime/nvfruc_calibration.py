"""NvFRUC end-to-end performance calibration and safe-limit helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any, Iterable


@dataclass(frozen=True)
class NvFrucCalibrationResult:
    max_output_fps: float
    safe_output_fps: int
    base_runtime_fps: int
    bottleneck: str
    passed: bool
    duration_s: float = 0.0
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def output_base_fps(output_target_fps: float, enabled: bool = True) -> int:
    target = max(1.0, float(output_target_fps))
    return max(1, int(math.ceil(target / 2.0))) if enabled else int(math.ceil(target))


def _p95(values: Iterable[float]) -> float:
    samples = [float(value) for value in values if float(value) > 0.0]
    if not samples:
        return 0.0
    if len(samples) < 2:
        return max(samples)
    return float(quantiles(samples, n=20, method="inclusive")[18])


def calculate_safe_output_limit(
    stage_times_ms: dict[str, Iterable[float]],
    *,
    output_cap_fps: float | None = None,
    safety_factor: float = 0.80,
) -> tuple[float, int, str]:
    """Calculate a sustainable output limit from p95 complete-chain stages."""
    candidates: list[tuple[str, float]] = []
    for name, samples in stage_times_ms.items():
        p95_ms = _p95(samples)
        if p95_ms > 0.0:
            candidates.append((str(name), 1000.0 / p95_ms))
    if output_cap_fps is not None and float(output_cap_fps) > 0.0:
        candidates.append(("output_cap", float(output_cap_fps)))
    if not candidates:
        return 0.0, 0, "insufficient_samples"
    bottleneck, max_fps = min(candidates, key=lambda item: item[1])
    safe = max(1, int(math.floor(max_fps * max(0.1, min(1.0, float(safety_factor))))))
    return float(max_fps), safe, bottleneck


def calibration_fingerprint(values: dict[str, Any]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class NvFrucCalibrationCache:
    """Small atomic JSON cache for writable per-configuration calibration data."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "nvfruc-calibration.json"

    def load(self, fingerprint: str) -> NvFrucCalibrationResult | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        try:
            return NvFrucCalibrationResult(
                max_output_fps=float(result["max_output_fps"]),
                safe_output_fps=int(result["safe_output_fps"]),
                base_runtime_fps=int(result["base_runtime_fps"]),
                bottleneck=str(result["bottleneck"]),
                passed=bool(result["passed"]),
                duration_s=float(result.get("duration_s", 0.0)),
                reason=result.get("reason"),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def save(self, fingerprint: str, result: NvFrucCalibrationResult) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": fingerprint, "result": result.as_dict()}
        fd, temp_name = tempfile.mkstemp(
            prefix=".nvfruc-calibration-",
            suffix=".json",
            dir=str(self.directory),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def downgrade_target_fps(
    current_target_fps: float,
    *,
    safety_factor: float = 0.90,
    minimum_fps: int = 1,
) -> int:
    return max(
        int(minimum_fps),
        int(math.floor(max(1.0, float(current_target_fps)) * float(safety_factor))),
    )


class NvFrucCalibrationController:
    """Three-phase complete-chain calibration state machine.

    The stage supplies measurements through :meth:`observe`; callers may also
    provide submission FPS, queue depth and complete-group drop counts from the
    selected output consumer.
    """

    def __init__(
        self,
        *,
        output_target_fps: int,
        fingerprint: str,
        cache: NvFrucCalibrationCache | None = None,
        clock=None,
        on_limit: Any | None = None,
        warmup_s: float = 3.0,
        measurement_s: float = 10.0,
        verification_s: float = 5.0,
    ) -> None:
        import time

        self.output_target_fps = max(1, int(output_target_fps))
        self.fingerprint = str(fingerprint)
        self.cache = cache
        self._clock = clock or time.monotonic
        self.on_limit = on_limit
        self.warmup_s = max(0.0, float(warmup_s))
        self.measurement_s = max(0.1, float(measurement_s))
        self.verification_s = max(0.1, float(verification_s))
        self.phase = "idle"
        self.started_at: float | None = None
        self._samples: dict[str, list[float]] = {}
        self._queue_samples: list[float] = []
        self._drop_samples: list[float] = []
        self.result: NvFrucCalibrationResult | None = None
        self.current_target_fps = self.output_target_fps

    @property
    def active(self) -> bool:
        return self.phase in {"warmup", "measure", "verify"}

    def start(self) -> NvFrucCalibrationResult | None:
        cached = self.cache.load(self.fingerprint) if self.cache else None
        if cached is not None and cached.passed:
            self.result = cached
            self.current_target_fps = min(self.output_target_fps, cached.safe_output_fps)
            self.phase = "cached"
            if self.on_limit is not None:
                self.on_limit(self.current_target_fps)
            return cached
        self.started_at = self._clock()
        self.phase = "warmup"
        self._samples.clear()
        self._queue_samples.clear()
        self._drop_samples.clear()
        self.result = None
        return None

    def _append(self, name: str, value: Any) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if numeric > 0.0:
            self._samples.setdefault(name, []).append(numeric)

    def observe(self, *, process_ms: float, submitted_fps: float = 0.0,
                queue_depth: float = 0.0, group_drops: float = 0.0,
                now: float | None = None) -> NvFrucCalibrationResult | None:
        if not self.active or self.started_at is None:
            return self.result
        timestamp = self._clock() if now is None else float(now)
        elapsed = timestamp - self.started_at
        if self.phase == "warmup":
            if elapsed < self.warmup_s:
                return None
            self.phase = "measure"
            self.started_at = timestamp
            self._samples.clear()
            self._queue_samples.clear()
            self._drop_samples.clear()
            return None
        self._append("nvfruc", process_ms)
        self._append("submitted", submitted_fps)
        self._queue_samples.append(max(0.0, float(queue_depth)))
        self._drop_samples.append(max(0.0, float(group_drops)))
        if self.phase == "measure" and elapsed >= self.measurement_s:
            maximum, safe, bottleneck = calculate_safe_output_limit(self._samples)
            self.result = NvFrucCalibrationResult(
                max_output_fps=maximum,
                safe_output_fps=min(self.output_target_fps, safe),
                base_runtime_fps=output_base_fps(min(self.output_target_fps, safe)),
                bottleneck=bottleneck,
                passed=False,
                duration_s=self.warmup_s + self.measurement_s,
                reason="verification_pending",
            )
            self.phase = "verify"
            self.started_at = timestamp
            self.current_target_fps = min(self.output_target_fps, safe)
            if self.on_limit is not None:
                self.on_limit(self.current_target_fps)
            return self.result
        if self.phase == "verify" and elapsed >= self.verification_s:
            dropped = sum(self._drop_samples)
            self.result = NvFrucCalibrationResult(
                max_output_fps=self.result.max_output_fps if self.result else 0.0,
                safe_output_fps=self.current_target_fps,
                base_runtime_fps=output_base_fps(self.current_target_fps),
                bottleneck=self.result.bottleneck if self.result else "unknown",
                passed=bool(dropped <= 0.0 and self.current_target_fps > 0),
                duration_s=self.warmup_s + self.measurement_s + self.verification_s,
                reason=None if dropped <= 0.0 else "complete_frame_group_drops",
            )
            self.phase = "complete"
            if self.cache and self.result.passed:
                self.cache.save(self.fingerprint, self.result)
            return self.result
        return self.result

    def downgrade(self) -> int:
        self.current_target_fps = downgrade_target_fps(self.current_target_fps)
        if self.on_limit is not None:
            self.on_limit(self.current_target_fps)
        return self.current_target_fps
