from __future__ import annotations

from xr_viewer.core_controller_guide_input import CoreControllerGuideInputMixin
from xr_viewer.core_controller_shortcuts import CoreControllerShortcutsMixin
from viewer.controller_help import get_controller_help_rows


class GuideHost(CoreControllerGuideInputMixin, CoreControllerShortcutsMixin):
    def __init__(self) -> None:
        self._frame_now = 1.0
        self._controller_inputs = ({}, {})
        self._keyboard_visible = False
        self._grip_target_l = None
        self._grip_target_r = None
        self._controller_calibration_mode = False
        self.actions: list[tuple[str, dict]] = []
        self._init_controller_shortcuts()
        self._init_controller_guide_input()

    def _dispatch_controller_shortcut(self, action: str, **values) -> None:
        self.actions.append((action, values))

    def update(self, *, left=None, right=None, after=0.0) -> None:
        self._frame_now += float(after)
        self._controller_inputs = (left or {}, right or {})
        self._handle_controller_guide_input(after)


def test_ab_chord_switches_brand_then_enters_calibration() -> None:
    host = GuideHost()
    buttons = {"a_button": 1.0, "b_button": 1.0}

    host.update(right=buttons)
    host.update(right=buttons, after=0.51)
    host.update(right=buttons, after=4.5)

    assert [action for action, _values in host.actions] == [
        "switch_controller_brand",
        "toggle_controller_calibration",
    ]


def test_grip_sticks_match_screen_and_depth_guide_rows() -> None:
    host = GuideHost()

    host.update(left={"grip": 1.0, "joystick_x": 0.5}, after=0.1)
    host.update(
        left={"joystick_y": 0.8},
        right={"grip": 1.0, "joystick_x": 0.6, "joystick_y": -0.4},
        after=0.1,
    )

    assert [action for action, _values in host.actions] == [
        "rotate_screen",
        "adjust_depth_strength",
        "resize_screen",
    ]


def test_no_grip_axes_and_keyboard_axes_are_exclusive() -> None:
    host = GuideHost()
    host.update(
        left={"joystick_x": 0.4, "joystick_y": -0.5},
        right={"joystick_y": 0.7},
        after=0.1,
    )
    host._keyboard_visible = True
    host._grip_target_l = "keyboard"
    host.update(
        left={"grip": 1.0, "stick_click": 1.0},
        right={"joystick_x": 0.5, "joystick_y": -0.5},
        after=0.1,
    )

    assert [action for action, _values in host.actions] == [
        "arrow_axes",
        "scroll_axes",
        "rotate_keyboard",
    ]


def test_keyboard_grip_controls_require_laser_latched_keyboard_target() -> None:
    host = GuideHost()
    host._keyboard_visible = True
    host._grip_target_l = "screen"
    host.update(
        left={"grip": 1.0, "stick_click": 1.0, "joystick_x": 0.5},
        after=0.1,
    )
    host._grip_target_l = "keyboard"
    host.update(
        left={"grip": 1.0, "stick_click": 1.0, "joystick_x": 0.5},
        after=0.1,
    )

    assert [action for action, _values in host.actions] == [
        "rotate_screen",
        "orbit_keyboard",
    ]


def test_calibration_axes_and_b_save_suppress_normal_controls() -> None:
    host = GuideHost()
    host._controller_calibration_mode = True

    host.update(
        left={"joystick_y": 0.5},
        right={"joystick_x": 0.4, "joystick_y": -0.3},
        after=0.1,
    )
    host.update(right={"b_button": 1.0}, after=0.1)

    assert [action for action, _values in host.actions] == [
        "adjust_controller_calibration",
        "save_controller_calibration",
    ]


def test_operation_guide_matches_b_long_press_product_contract() -> None:
    cn_rows, _cn_environment_rows = get_controller_help_rows("CN")
    en_rows, _en_environment_rows = get_controller_help_rows("EN")

    assert ("右 B 键", "长按 1s", "显示/隐藏操作指南", False) in cn_rows
    assert (
        "Right B button",
        "Long press 1s",
        "Show/hide operation guide",
        False,
    ) in en_rows
