from __future__ import annotations

import time


class CoreControllerShortcutsMixin:
    """Renderer-independent legacy controller shortcut state machine."""

    _SHORTCUT_LONG_SECONDS = 1.0
    _MENU_SHORT_MAX_SECONDS = 0.6
    _X_PASSTHROUGH_SECONDS = 4.0

    def _init_controller_shortcuts(self) -> None:
        self._shortcut_last = {
            "menu": False,
            "a": False,
            "b": False,
            "x": False,
            "y": False,
            "left_stick": False,
            "right_stick": False,
        }
        self._shortcut_pressed_at = {name: 0.0 for name in self._shortcut_last}
        self._shortcut_long_fired = {name: False for name in self._shortcut_last}

    def _shortcut_time(self) -> float:
        frame_now = float(getattr(self, "_frame_now", 0.0) or 0.0)
        return frame_now if frame_now > 0.0 else time.perf_counter()

    def _emit_controller_shortcut(self, action: str) -> None:
        dispatcher = getattr(self, "_dispatch_controller_shortcut", None)
        if callable(dispatcher):
            dispatcher(action)

    @staticmethod
    def _shortcut_pressed(hand: dict[str, float], name: str) -> bool:
        return float(hand.get(name, 0.0) or 0.0) > 0.5

    def _update_short_long_button(
        self,
        name: str,
        pressed: bool,
        now: float,
        *,
        short_action: str,
        long_action: str,
        enabled: bool = True,
    ) -> None:
        was_pressed = self._shortcut_last[name]
        if pressed and not was_pressed:
            self._shortcut_pressed_at[name] = now
            self._shortcut_long_fired[name] = not enabled
        if pressed and not enabled:
            self._shortcut_long_fired[name] = True
        if enabled and pressed and not self._shortcut_long_fired[name]:
            if now - self._shortcut_pressed_at[name] >= self._SHORTCUT_LONG_SECONDS:
                self._emit_controller_shortcut(long_action)
                self._shortcut_long_fired[name] = True
        if enabled and not pressed and was_pressed and not self._shortcut_long_fired[name]:
            self._emit_controller_shortcut(short_action)
        self._shortcut_last[name] = pressed

    def _update_x_shortcuts(self, pressed: bool, now: float) -> None:
        name = "x"
        was_pressed = self._shortcut_last[name]
        if pressed and not was_pressed:
            self._shortcut_pressed_at[name] = now
            self._shortcut_long_fired[name] = False
        if pressed and not self._shortcut_long_fired[name]:
            if now - self._shortcut_pressed_at[name] >= self._X_PASSTHROUGH_SECONDS:
                self._emit_controller_shortcut("toggle_passthrough")
                self._shortcut_long_fired[name] = True
        if not pressed and was_pressed and not self._shortcut_long_fired[name]:
            held = now - self._shortcut_pressed_at[name]
            self._emit_controller_shortcut(
                "cycle_environment_light"
                if held >= self._SHORTCUT_LONG_SECONDS
                else "toggle_keyboard"
            )
        self._shortcut_last[name] = pressed

    def _update_stick_shortcut(
        self,
        hand: str,
        pressed: bool,
        grip_pressed: bool,
        now: float,
    ) -> None:
        name = f"{hand}_stick"
        was_pressed = self._shortcut_last[name]
        if pressed and not was_pressed:
            self._shortcut_pressed_at[name] = now
            self._shortcut_long_fired[name] = grip_pressed
            if grip_pressed:
                self._emit_controller_shortcut(
                    "toggle_stereo" if hand == "left" else "reset_depth"
                )
        if pressed and not grip_pressed and not self._shortcut_long_fired[name]:
            if now - self._shortcut_pressed_at[name] >= self._SHORTCUT_LONG_SECONDS:
                self._emit_controller_shortcut("cut" if hand == "left" else "enter")
                self._shortcut_long_fired[name] = True
        if not pressed and was_pressed and not self._shortcut_long_fired[name]:
            self._emit_controller_shortcut("copy" if hand == "left" else "paste")
        self._shortcut_last[name] = pressed

    def _handle_controller_shortcuts(self) -> None:
        """Translate controller snapshots into legacy semantic actions."""
        left, right = self._controller_inputs
        now = self._shortcut_time()
        pressed = self._shortcut_pressed

        menu = pressed(left, "menu_button") or pressed(right, "menu_button")
        menu_was_pressed = self._shortcut_last["menu"]
        if menu and not menu_was_pressed:
            self._shortcut_pressed_at["menu"] = now
        if not menu and menu_was_pressed:
            if now - self._shortcut_pressed_at["menu"] < self._MENU_SHORT_MAX_SECONDS:
                self._emit_controller_shortcut("cycle_status_panel")
        self._shortcut_last["menu"] = menu

        a = pressed(right, "a_button")
        b = pressed(right, "b_button")
        right_grip = pressed(right, "grip")
        normal_ab = (
            not (a and b)
            and not right_grip
            and not bool(getattr(self, "_controller_calibration_mode", False))
        )
        self._update_short_long_button(
            "a", a, now,
            short_action="toggle_screen_shape",
            long_action="cycle_status_panel",
            enabled=normal_ab,
        )
        self._update_short_long_button(
            "b", b, now,
            short_action="toggle_background",
            long_action="cycle_hand_panel",
            enabled=normal_ab,
        )
        self._update_short_long_button(
            "y", pressed(left, "y_button"), now,
            short_action="reset_screen",
            long_action="cycle_screen_preset",
        )
        self._update_x_shortcuts(pressed(left, "x_button"), now)

        left_stick = pressed(left, "stick_click")
        left_stick_reserved = (
            left_stick
            and pressed(left, "grip")
            and bool(getattr(self, "_keyboard_visible", False))
            and getattr(self, "_grip_target_l", None) == "keyboard"
        )
        if left_stick_reserved:
            self._shortcut_last["left_stick"] = True
            self._shortcut_long_fired["left_stick"] = True
        else:
            self._update_stick_shortcut(
                "left", left_stick, pressed(left, "grip"), now
            )
        self._update_stick_shortcut(
            "right", pressed(right, "stick_click"), right_grip, now
        )
