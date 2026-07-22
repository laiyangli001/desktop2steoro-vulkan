from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_runtime.runtime_callbacks import RuntimeCallbacks


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
    )
    return RuntimeCallbacks(context)


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
