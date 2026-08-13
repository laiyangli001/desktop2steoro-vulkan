from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class MenuControl:
    key: str
    label: str
    rect: tuple[float, float, float, float]
    kind: str = "button"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    enabled: bool = True

    def contains(self, u: float, v: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= u <= x1 and y0 <= v <= y1

    def value_from_u(self, u: float) -> float:
        x0, _y0, x1, _y1 = self.rect
        fraction = max(0.0, min(1.0, (float(u) - x0) / max(x1 - x0, 1e-9)))
        raw = self.minimum + fraction * (self.maximum - self.minimum)
        steps = round((raw - self.minimum) / max(self.step, 1e-9))
        return max(self.minimum, min(self.maximum, self.minimum + steps * self.step))


PICTURE_CONTROLS = (
    ("color_brightness", "Brightness", 0.2, 2.0, 0.1),
    ("color_contrast", "Contrast", 0.5, 2.0, 0.1),
    ("color_saturation", "Saturation", 0.0, 2.0, 0.1),
    ("color_gamma", "Gamma", 0.5, 2.0, 0.1),
    ("color_temperature", "Temperature", -100.0, 100.0, 10.0),
    ("color_tint", "Tint", -100.0, 100.0, 10.0),
    ("vulkan_projection_min_lod", "Min LOD", 0.0, 2.0, 0.05),
    ("vulkan_projection_max_lod", "Max LOD", 0.0, 2.0, 0.05),
    ("vulkan_projection_mip_lod_bias", "MIP Bias", -1.5, 0.0, 0.05),
    ("vulkan_projection_rcas_sharpness", "RCAS", 0.0, 1.0, 0.05),
)


class OpenXrSettingsMenu:
    """Renderer-independent tab, hit-test, and trigger state for the XR menu."""

    tabs = ("picture", "room", "screen")

    def __init__(self) -> None:
        self.visible = False
        self.tab = "picture"
        self.hover_key: str | None = None
        self.active_hand: int | None = None
        self.active_key: str | None = None
        self.dirty = True
        self.revision = 0
        self._trigger_down = [False, False]
        self._outside_down = [False, False]

    def open(self) -> None:
        self.visible = True
        self.active_hand = None
        self.active_key = None
        self.mark_dirty()

    def close(self) -> None:
        self.visible = False
        self.hover_key = None
        self.active_hand = None
        self.active_key = None
        self.mark_dirty()

    def mark_dirty(self) -> None:
        self.dirty = True
        self.revision += 1

    def set_tab(self, tab: str) -> bool:
        if tab not in self.tabs or tab == self.tab:
            return False
        self.tab = tab
        self.hover_key = None
        self.mark_dirty()
        return True

    def controls(self, *, allow_curve: bool = True) -> tuple[MenuControl, ...]:
        controls = [
            MenuControl("tab:picture", "Picture", (0.05, 0.055, 0.32, 0.13)),
            MenuControl("tab:room", "Room", (0.365, 0.055, 0.635, 0.13)),
            MenuControl("tab:screen", "Screen", (0.68, 0.055, 0.95, 0.13)),
            MenuControl("close", "Close", (0.91, 0.015, 0.975, 0.05)),
        ]
        if self.tab == "picture":
            for index, (key, label, minimum, maximum, step) in enumerate(PICTURE_CONTROLS):
                column, row = divmod(index, 5)
                y0 = 0.19 + row * 0.145
                x0 = 0.08 + column * 0.47
                controls.append(MenuControl(key, label, (x0, y0 + 0.055, x0 + 0.38, y0 + 0.105), "slider", minimum, maximum, step))
        elif self.tab == "room":
            controls.extend((
                MenuControl("room:previous_seat", "Previous seat", (0.08, 0.24, 0.44, 0.34)),
                MenuControl("room:next_seat", "Next seat", (0.56, 0.24, 0.92, 0.34)),
                MenuControl("room:seat_height", "Seat height", (0.12, 0.49, 0.88, 0.56), "slider", -1.0, 1.0, 0.02),
                MenuControl("room:exposure", "Scene brightness", (0.12, 0.69, 0.88, 0.76), "slider", -4.0, 4.0, 0.1),
            ))
        else:
            controls.extend((
                MenuControl("screen:width", "Screen size", (0.12, 0.27, 0.88, 0.34), "slider", 0.25, 2.0, 0.01),
                MenuControl("screen:height", "Screen height", (0.12, 0.49, 0.88, 0.56), "slider", -2.0, 2.0, 0.02),
                MenuControl("screen:curve", "Curved screen", (0.30, 0.70, 0.70, 0.81), enabled=allow_curve),
            ))
        return tuple(controls)

    def hit_test(self, uv: tuple[float, float] | None, *, allow_curve: bool = True) -> MenuControl | None:
        if uv is None:
            return None
        u, v = uv
        for control in reversed(self.controls(allow_curve=allow_curve)):
            if control.enabled and control.contains(float(u), float(v)):
                return control
        return None

    def sample_trigger(self, hand: int, value: float, *, outside_targets: bool) -> bool:
        """Return True once a short outside click should open the menu."""
        hand = int(hand)
        pressed = float(value) >= 0.7
        released = float(value) <= 0.3
        if not self._trigger_down[hand] and pressed:
            self._trigger_down[hand] = True
            self._outside_down[hand] = bool(outside_targets)
        elif self._trigger_down[hand] and released:
            should_open = self._outside_down[hand] and bool(outside_targets)
            self._trigger_down[hand] = False
            self._outside_down[hand] = False
            return should_open
        return False


def clamp_picture_values(values: dict[str, float]) -> dict[str, float]:
    result = dict(values)
    if "vulkan_projection_min_lod" in result and "vulkan_projection_max_lod" in result:
        result["vulkan_projection_min_lod"] = min(
            float(result["vulkan_projection_min_lod"]),
            float(result["vulkan_projection_max_lod"]),
        )
    return result


def control_by_key(controls: Iterable[MenuControl], key: str) -> MenuControl | None:
    return next((control for control in controls if control.key == key), None)
