"""Shared startup splash for the legacy and GUI2 Flet shells."""

from __future__ import annotations

import math
from pathlib import Path

import flet as ft

from .paths import BASE_DIR


STARTUP_IMAGE_PATH = Path(BASE_DIR) / "d2s_blur.png"
STARTUP_IMAGE_ASPECT = 1672 / 941


def startup_window_size(display_size: tuple[int, int]) -> tuple[int, int]:
    """Return a proportional splash size covering about a quarter display."""
    display_width, display_height = (max(1, int(value)) for value in display_size)
    target_area = display_width * display_height * 0.25
    width = math.sqrt(target_area * STARTUP_IMAGE_ASPECT)
    height = width / STARTUP_IMAGE_ASPECT
    scale = min(1.0, (display_width * 0.9) / width, (display_height * 0.9) / height)
    return max(240, round(width * scale)), max(135, round(height * scale))


def _primary_display_size() -> tuple[int, int]:
    """Read the primary display dimensions for splash sizing only."""
    try:
        import mss

        with mss.mss() as screenshot:
            monitor = screenshot.monitors[1]
            return int(monitor["width"]), int(monitor["height"])
    except Exception:
        return 1920, 1080


async def configure_startup_splash(page: ft.Page) -> None:
    """Show the centered splash until the application mounts its real UI."""
    # Resolve the display geometry before creating the native window so the
    # first frame and the image use exactly the same aspect ratio and size.
    display_width, display_height = _primary_display_size()
    width, height = startup_window_size((display_width, display_height))
    # Frameless is limited to the short startup surface. The full application
    # restores normal window chrome before it becomes interactive.
    page.window.frameless = True
    page.window.width = width
    page.window.height = height
    page.padding = 0
    if STARTUP_IMAGE_PATH.is_file():
        content = ft.Image(
            src=str(STARTUP_IMAGE_PATH),
            fit=ft.BoxFit.CONTAIN,
            expand=True,
        )
    else:
        content = ft.Container(bgcolor=ft.Colors.BLACK, expand=True)
    page.add(
        ft.Container(
            content=content,
            expand=True,
            alignment=ft.Alignment.CENTER,
            bgcolor=ft.Colors.BLACK,
        )
    )
    page.window.visible = True
    page.update()
