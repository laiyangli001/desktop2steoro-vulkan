"""Small declarative menu registry used by GUI2.

The registry keeps future menu additions out of the root layout builder while
leaving callbacks owned by the GUI instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass(frozen=True)
class MenuItemSpec:
    item_id: str
    label_key: str
    callback_name: str | None = None
    icon: Any = None
    children: tuple["MenuItemSpec", ...] = field(default_factory=tuple)
    enabled: bool = True


def build_menu_specs() -> tuple[MenuItemSpec, ...]:
    """Return the initial menu tree in stable, testable order."""
    return (
        MenuItemSpec(
            "settings",
            "menu_settings",
            children=(
                MenuItemSpec("reset_defaults", "menu_reset_defaults", "confirm_reset_defaults"),
            ),
        ),
        MenuItemSpec(
            "tools",
            "menu_tools",
            children=(MenuItemSpec("diagnostics", "menu_diagnostics", "open_diagnostics"),),
        ),
    )
