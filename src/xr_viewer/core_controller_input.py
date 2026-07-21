"""OpenXR controller input values used by Filament controller animations."""

from __future__ import annotations


class CoreControllerInputMixin:
    def _read_bool_action(self, action, hand_path):
        if action is None:
            return False
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_boolean(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return bool(state.is_active and state.current_state)
        except Exception:
            return False

    def _read_float_action(self, action, hand_path):
        if action is None:
            return 0.0
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_float(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return float(state.current_state) if state.is_active else 0.0
        except Exception:
            return 0.0

    def _read_stick_action(self, action, hand_path):
        if action is None:
            return 0.0, 0.0
        try:
            path = self._path_left if hand_path == "/user/hand/left" else self._path_right
            state = self.xr.get_action_state_vector2f(
                self.session,
                self.xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            if not state.is_active:
                return 0.0, 0.0
            return float(state.current_state.x), -float(state.current_state.y)
        except Exception:
            return 0.0, 0.0

    def _sync_controller_inputs(self, delta_seconds: float) -> None:
        self.xr.sync_actions(self.session, self._xr_actions_sync_info)
        lx, ly = self._read_stick_action(self._act_left_stick, "/user/hand/left")
        rx, ry = self._read_stick_action(self._act_right_stick, "/user/hand/right")

        def values(hand: str, left: bool) -> dict[str, float]:
            prefix = "left_" if left else "right_"
            trigger = self._read_float_action(
                self._act_left_trigger if left else self._act_right_trigger, hand
            )
            grip = 1.0 if self._read_bool_action(
                self._act_left_grip if left else self._act_right_grip, hand
            ) else 0.0
            buttons = {
                "a_button": self._act_a_btn,
                "b_button": self._act_b_btn,
                "x_button": self._act_x_btn,
                "y_button": self._act_y_btn,
                "menu_button": self._act_menu_btn,
            }
            result = {"trigger": trigger, "grip": grip}
            for name, action in buttons.items():
                result[name] = 1.0 if self._read_bool_action(action, hand) else 0.0
            return result

        left = values("/user/hand/left", True)
        right = values("/user/hand/right", False)
        left["stick_click"] = 1.0 if self._read_bool_action(self._act_left_stick_click, "/user/hand/left") else 0.0
        right["stick_click"] = 1.0 if self._read_bool_action(self._act_right_stick_click, "/user/hand/right") else 0.0
        left.update({"joystick_x": lx, "joystick_y": ly, "touchpad_x": lx, "touchpad_y": ly})
        right.update({"joystick_x": rx, "joystick_y": ry, "touchpad_x": rx, "touchpad_y": ry})
        self._controller_inputs = (left, right)

    def _controller_input(self, hand: int) -> dict[str, float]:
        values = getattr(self, "_controller_inputs", ({}, {}))
        return values[0 if int(hand) == 0 else 1]
