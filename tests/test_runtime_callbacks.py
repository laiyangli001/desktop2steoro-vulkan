from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_runtime.runtime_callbacks import RuntimeCallbacks
from stereo_runtime.settings_snapshot import RuntimeSettingsSnapshot


class FakeOpenXrState:
    def __init__(self, depth_strength: float) -> None:
        self.runtime_settings_snapshot = SimpleNamespace(
            depth_strength=depth_strength
        )
        self.updates: list[dict[str, float | None]] = []

    def update_runtime_config(self, **values) -> None:
        self.updates.append(values)
        if values.get("depth_strength") is not None:
            self.runtime_settings_snapshot.depth_strength = values[
                "depth_strength"
            ]


def _callbacks(depth_strength: float = 0.75) -> RuntimeCallbacks:
    context = SimpleNamespace(
        stereo_runtime=SimpleNamespace(
            stereo_config=SimpleNamespace(depth_strength=depth_strength)
        ),
        openxr_state=FakeOpenXrState(depth_strength),
        fps_breakdown=SimpleNamespace(inc=lambda *_args, **_kwargs: None),
    )
    return RuntimeCallbacks(context)


def test_capture_fps_reports_accepted_capture_rate_and_expires(monkeypatch) -> None:
    callbacks = _callbacks()
    times = iter((10.0, 10.5, 11.0))
    monkeypatch.setattr(
        "app_runtime.runtime_callbacks.time.perf_counter",
        lambda: next(times),
    )

    callbacks.breakdown_inc("capture")
    callbacks.breakdown_inc("capture")

    assert callbacks.capture_fps() == pytest.approx(2.0)

    monkeypatch.setattr(
        "app_runtime.runtime_callbacks.time.perf_counter", lambda: 12.0
    )
    assert callbacks.capture_fps() == 0.0


def test_controller_shortcut_toggles_stereo_and_restores_depth() -> None:
    callbacks = _callbacks(0.75)

    assert callbacks.on_openxr_controller_shortcut("toggle_stereo") is True
    assert callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength == 0.0

    assert callbacks.on_openxr_controller_shortcut("toggle_stereo") is True
    assert (
        callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength
        == pytest.approx(0.75)
    )


def test_controller_shortcut_resets_depth_and_rejects_renderer_action() -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength = 0.2

    assert callbacks.on_openxr_controller_shortcut("reset_depth") is True
    assert (
        callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength
        == pytest.approx(0.6)
    )
    assert callbacks.on_openxr_controller_shortcut("toggle_screen_shape") is False


def test_controller_shortcut_adjusts_depth_continuously_with_clamp() -> None:
    callbacks = _callbacks(0.6)

    assert callbacks.on_openxr_controller_shortcut(
        "adjust_depth_strength", delta=0.25
    ) is True
    assert (
        callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength
        == pytest.approx(0.85)
    )
    callbacks.on_openxr_controller_shortcut(
        "adjust_depth_strength", delta=-20.0
    )
    assert callbacks.context.openxr_state.runtime_settings_snapshot.depth_strength == 0.0


def test_settings_menu_runtime_value_updates_snapshot_without_persisting() -> None:
    callbacks = _callbacks(0.6)
    callbacks.context.openxr_state.runtime_settings_snapshot = RuntimeSettingsSnapshot(
        version=4, timestamp=1.0, depth_strength=0.6
    )
    snapshots = []
    callbacks.context.settings_update_q = SimpleNamespace(
        put_nowait=lambda snapshot: snapshots.append(snapshot),
        get_nowait=lambda: (_ for _ in ()).throw(Exception()),
    )
    callbacks.send_settings_snapshot = snapshots.append

    assert callbacks.on_openxr_controller_shortcut(
        "set_runtime_setting", name="color_brightness", value=1.4,
        persist=False,
    ) is True
    snapshot = callbacks.context.openxr_state.updates[-1]["snapshot"]
    assert snapshot.version == 5
    assert snapshot.color_brightness == pytest.approx(1.4)
    assert snapshots[-1] is snapshot
