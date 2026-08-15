"""Resolution-aware scaling math for the Flet desktop GUI."""

from __future__ import annotations


REFERENCE_DISPLAY_WIDTH = 3840.0
REFERENCE_DISPLAY_HEIGHT = 2160.0
MIN_UI_SCALE = 0.5
MAX_UI_SCALE = 2.0
REFERENCE_DEVICE_PIXEL_RATIO = 2.0


def clamp_ui_scale(value: float) -> float:
    """Clamp a UI scale to the supported automatic range."""
    return max(MIN_UI_SCALE, min(MAX_UI_SCALE, float(value)))


def compute_display_scale(width: float | None, height: float | None) -> float:
    """Return the startup scale for a physical display resolution.

    The current 4K layout is the 100% design reference.  The smaller axis is
    used so unusual aspect ratios cannot make the window overflow vertically.
    """
    try:
        width_value = float(width or 0)
        height_value = float(height or 0)
    except (TypeError, ValueError):
        return 1.0
    if width_value <= 0 or height_value <= 0:
        return 1.0
    return clamp_ui_scale(
        min(
            width_value / REFERENCE_DISPLAY_WIDTH,
            height_value / REFERENCE_DISPLAY_HEIGHT,
        )
    )


def compute_dpi_compensated_display_scale(
    width: float | None,
    height: float | None,
    device_pixel_ratio: float | None,
) -> float:
    """Return the transform scale after accounting for platform DPI.

    The 4K reference UI is observed at Windows 200% scaling.  Flet already
    applies the monitor DPR to logical controls, so the explicit resolution
    transform must compensate for that platform scaling.  Otherwise moving
    from 4K/200% to 1080p/100% applies both changes and shrinks the UI twice.
    """
    try:
        dpr = max(float(device_pixel_ratio or 1.0), 0.1)
    except (TypeError, ValueError):
        dpr = 1.0
    physical_scale = compute_display_scale(width, height)
    return clamp_ui_scale(
        physical_scale * REFERENCE_DEVICE_PIXEL_RATIO / dpr
    )


def compute_viewport_scale(
    viewport_width: float | None,
    base_viewport_width: float | None,
    display_scale: float,
) -> float:
    """Return the width-only multiplier relative to the display scale."""
    try:
        viewport = float(viewport_width or 0)
        base = float(base_viewport_width or 0)
        display = max(float(display_scale), 1e-6)
    except (TypeError, ValueError):
        return 1.0
    if viewport <= 0 or base <= 0:
        return 1.0
    return viewport / (base * display)


def compute_effective_ui_scale(
    viewport_width: float | None,
    base_viewport_width: float | None,
    display_scale: float,
) -> tuple[float, float]:
    """Return ``(viewport_scale, effective_scale)`` for a page width.

    Device-pixel ratio is intentionally absent: Flet already reports the page
    width in logical pixels and maps those pixels through the platform DPR.
    """
    viewport_scale = compute_viewport_scale(
        viewport_width,
        base_viewport_width,
        display_scale,
    )
    effective_scale = clamp_ui_scale(float(display_scale) * viewport_scale)
    return viewport_scale, effective_scale
