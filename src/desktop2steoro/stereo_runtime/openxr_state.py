from __future__ import annotations

import threading
from dataclasses import replace

from .adapter import openxr_render_config_from_snapshot
from .openxr_render import OpenXRRenderConfig
from .settings_snapshot import RuntimeSettingsSnapshot


class OpenXRStateController:
    def __init__(
        self,
        *,
        run_mode: str,
        depth_ratio: float | None = None,
        depth_strength: float | None = None,
        convergence: float = 0.0,
    ):
        self.run_mode = run_mode
        self.render_active = threading.Event()
        self.source_active = threading.Event()
        self.wait_idle_active = threading.Event()
        self.bootstrap_done = threading.Event()
        self.runtime_config_lock = threading.Lock()
        if depth_ratio is None:
            depth_ratio = depth_strength
        if depth_ratio is None:
            raise TypeError("OpenXRStateController requires depth_ratio or depth_strength")
        self.runtime_settings_snapshot = RuntimeSettingsSnapshot(
            version=0,
            timestamp=0.0,
            depth_strength=float(depth_ratio),
            convergence=float(convergence),
        )
        self.screen_roll = 0.0
        self.source_pause_notice_lock = threading.Lock()
        self.source_pause_noticed = False
        self.wait_idle_notice_lock = threading.Lock()
        self.wait_idle_noticed = False

    def source_paused(self) -> bool:
        paused = (
            self.run_mode == "OpenXR"
            and self.bootstrap_done.is_set()
            and not self.source_active.is_set()
        )
        with self.source_pause_notice_lock:
            if self.source_pause_noticed is not paused:
                self.source_pause_noticed = paused
                if paused:
                    print("[Main] OpenXR source gate closed")
                else:
                    print("[Main] OpenXR source gate opened; waiting for runtime frame")
        return paused

    def hard_idle_active(self, on_enter=None) -> bool:
        idle = (
            self.run_mode == "OpenXR"
            and self.bootstrap_done.is_set()
            and self.wait_idle_active.is_set()
        )
        with self.wait_idle_notice_lock:
            if self.wait_idle_noticed is not idle:
                self.wait_idle_noticed = idle
                if idle:
                    if on_enter is not None:
                        on_enter()
                    print("[Main] OpenXR hard idle entered")
                else:
                    print("[Main] OpenXR hard idle exited; waiting for source gate")
        return idle

    def update_runtime_config(
        self,
        *,
        snapshot=None,
        depth_ratio=None,
        depth_strength=None,
        convergence=None,
        max_disparity_px=None,
        parallax_preset=None,
        screen_roll=None,
    ) -> None:
        with self.runtime_config_lock:
            current = self.runtime_settings_snapshot
            if snapshot is not None:
                current = _merge_snapshot(current, snapshot)
            if depth_ratio is None:
                depth_ratio = depth_strength
            if depth_ratio is not None:
                current = replace(current, depth_strength=float(depth_ratio))
            if convergence is not None:
                current = replace(current, convergence=float(convergence))
            if max_disparity_px is not None:
                current = replace(current, max_disparity_px=float(max_disparity_px))
            if parallax_preset is not None:
                current = replace(current, parallax_preset=str(parallax_preset))
            if screen_roll is not None:
                self.screen_roll = float(screen_roll)
            self.runtime_settings_snapshot = current

    def current_render_config(self, runtime) -> OpenXRRenderConfig:
        with self.runtime_config_lock:
            snapshot = _snapshot_with_runtime_fallbacks(self.runtime_settings_snapshot, runtime)
            screen_roll = self.screen_roll
        config = openxr_render_config_from_snapshot(
            snapshot,
            preset=getattr(runtime.stereo_config, "parallax_preset", "standard"),
            screen_roll=screen_roll,
        )
        return config


def _merge_snapshot(base: RuntimeSettingsSnapshot, updates: RuntimeSettingsSnapshot) -> RuntimeSettingsSnapshot:
    values = {}
    for name in updates.__dataclass_fields__:
        value = getattr(updates, name)
        if value is not None:
            values[name] = value
    if values:
        return replace(base, **values)
    return base


def _snapshot_with_runtime_fallbacks(snapshot: RuntimeSettingsSnapshot, runtime) -> RuntimeSettingsSnapshot:
    stereo_config = runtime.stereo_config
    return replace(
        snapshot,
        max_disparity_px=(
            getattr(stereo_config, "max_disparity_px", None)
            if snapshot.max_disparity_px is None
            else snapshot.max_disparity_px
        ),
        parallax_preset=(
            getattr(stereo_config, "parallax_preset", "standard")
            if snapshot.parallax_preset is None
            else snapshot.parallax_preset
        ),
    )
