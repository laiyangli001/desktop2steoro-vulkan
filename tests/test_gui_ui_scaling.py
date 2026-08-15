import asyncio
import time
from types import SimpleNamespace

import flet as ft

from gui.ui_scaling import (
    compute_dpi_compensated_display_scale,
    compute_display_scale,
    compute_effective_ui_scale,
    compute_viewport_scale,
)
from gui.controls import FONT_SIZE, S, CompactDropdown
from gui.gui import Desktop2StereoGUI


class _FakeWindow(SimpleNamespace):
    def update(self):
        pass


class _FakePage(SimpleNamespace):
    def add(self, control):
        self.controls.append(control)

    def update(self):
        pass


def _make_gui():
    page = _FakePage(
        controls=[],
        padding=0,
        width=1920,
        height=1080,
        window=_FakeWindow(
            left=0,
            top=0,
            width=None,
            height=None,
            min_width=None,
            min_height=None,
            max_width=None,
        ),
    )
    app = Desktop2StereoGUI(page)
    app.build_ui()
    return app


def test_display_scale_uses_4k_as_100_percent():
    assert compute_display_scale(1920, 1080) == 0.5
    assert compute_display_scale(3840, 2160) == 1.0
    assert compute_display_scale(7680, 4320) == 2.0


def test_display_scale_clamps_and_handles_invalid_detection():
    assert compute_display_scale(1280, 720) == 0.5
    assert compute_display_scale(15360, 8640) == 2.0
    assert compute_display_scale(None, None) == 1.0
    assert compute_display_scale(0, 2160) == 1.0


def test_unusual_display_aspect_ratio_uses_smaller_axis():
    assert compute_display_scale(3840, 1080) == 0.5
    assert compute_display_scale(1920, 2160) == 0.5


def test_viewport_scaling_is_width_only():
    assert compute_viewport_scale(750, 1000, 0.5) == 1.5
    viewport_scale, effective_scale = compute_effective_ui_scale(750, 1000, 0.5)
    assert viewport_scale == 1.5
    assert effective_scale == 0.75


def test_effective_scale_does_not_include_device_pixel_ratio():
    first = compute_effective_ui_scale(1000, 1000, 1.0)
    second = compute_effective_ui_scale(1000, 1000, 1.0)
    assert first == second == (1.0, 1.0)


def test_display_transform_compensates_for_platform_dpi_mapping():
    assert compute_dpi_compensated_display_scale(3840, 2160, 2.0) == 1.0
    assert compute_dpi_compensated_display_scale(1920, 1080, 1.0) == 1.0
    assert compute_dpi_compensated_display_scale(7680, 4320, 2.0) == 2.0


def test_display_scale_uses_physical_monitor_resolution_independent_of_dpr():
    app = _make_gui()
    app._display_resolution_for_window = lambda: (3840, 2160)

    app.device_pixel_ratio = 2.0
    assert app._refresh_display_scale() == 1.0


def test_window_monitor_resolution_uses_native_point_lookup_before_mss_rectangles(monkeypatch):
    app = _make_gui()
    app.page.window.left = 100
    app.page.window.top = 50
    app.page.width = 800
    app.page.height = 600
    monitors = [
        {
            "capture_index": 1,
            "display_number": 1,
            "left": 0,
            "top": 0,
            "width": 3840,
            "height": 2160,
        },
        {
            "capture_index": 2,
            "display_number": 2,
            "left": 3840,
            "top": 0,
            "width": 1920,
            "height": 1080,
        },
    ]
    monkeypatch.setattr("gui.builders.list_monitors", lambda: monitors)
    calls = []
    monkeypatch.setattr(
        app,
        "_windows_physical_resolution_at_point",
        lambda x, y: calls.append((x, y)) or (1920, 1080),
    )

    assert app._display_resolution_for_window() == (1920, 1080)
    assert calls == [(500.0, 350.0)]


def test_window_moved_event_waits_for_new_monitor_dpr_before_refresh():
    async def run_scenario():
        app = _make_gui()
        app._loop = asyncio.get_running_loop()
        calls = []
        app._on_page_media_change = lambda e=None: calls.append(e)

        app._on_window_event(SimpleNamespace(type=ft.WindowEventType.MOVED))
        assert calls == []
        assert app._ui_monitor_transition_until > 0.0

        await asyncio.sleep(0.35)
        assert calls == [None]

    asyncio.run(run_scenario())


def test_resize_during_monitor_transition_preserves_current_scale():
    app = _make_gui()
    app.effective_ui_scale = 0.8
    app.viewport_scale = 0.8
    app._ui_monitor_transition_until = time.monotonic() + 1.0

    app._on_page_resize(SimpleNamespace(width=700.0, height=600.0))

    assert app.effective_ui_scale == 0.8
    assert app.viewport_scale == 0.8


def test_media_change_preserves_runtime_scale_across_displays():
    async def run_scenario():
        app = _make_gui()
        app.device_pixel_ratio = 1.0
        app.display_scale = 0.75
        app.viewport_scale = 0.8
        app.effective_ui_scale = 0.6

        await app._apply_media_change()
        assert app.display_scale == 0.75
        assert app.viewport_scale == 0.8
        assert app.effective_ui_scale == 0.6

    asyncio.run(run_scenario())


def test_control_triggered_window_refit_does_not_change_monitor_scale():
    app = _make_gui()
    app.display_scale = 0.5
    app.viewport_scale = 1.0
    app.effective_ui_scale = 0.5
    app.device_pixel_ratio = 2.0
    app._display_resolution_for_window = lambda: (3840, 2160)

    app._fit_window_to_content(update=False, resize_window=False)

    assert app.display_scale == 0.5
    assert app.viewport_scale == 1.0
    assert app.effective_ui_scale == 0.5


def test_viewport_scale_clamps_only_the_effective_result():
    viewport_scale, effective_scale = compute_effective_ui_scale(5000, 1000, 1.0)
    assert viewport_scale == 5.0
    assert effective_scale == 2.0


def test_compact_dropdown_scales_popup_route_without_resizing_trigger():
    dropdown = CompactDropdown(
        options=["Short", "Long popup option"],
        value="Short",
        min_width=S(100),
    )
    trigger_height = dropdown.height

    dropdown.set_overlay_scale(0.5)

    first_item = dropdown.content.items[0]
    assert dropdown.height == trigger_height
    assert first_item.height == S(32) * 0.5
    assert first_item.content.content.size == FONT_SIZE * 0.5
    assert first_item.content.width >= S(100) * 0.5


def test_left_settings_column_uses_remaining_height_and_explicit_scrollbar():
    app = _make_gui()

    assert app._scroll_area.expand is True
    assert app._scroll_area.tight is False
    assert isinstance(app._scroll_area.scroll, ft.Scrollbar)
    assert app._scroll_area.scroll.orientation == ft.ScrollbarOrientation.RIGHT


def test_footer_follows_settings_in_scroll_content_flow():
    app = _make_gui()

    assert app._settings_content.controls[-1] is app._footer
    assert app._main_panel.content is app._scroll_area


def test_native_window_size_stays_logical_and_caps_to_logical_display():
    app = _make_gui()
    app.device_pixel_ratio = 2.0
    app.display_scale = 1.0
    app.effective_ui_scale = 1.0
    app._display_resolution_for_window = lambda: (3840, 2160)
    expected_base_width = app._estimate_window_width(app._estimate_main_panel_width())

    app._fit_window_to_content(update=False, resize_window=True)

    assert app._ui_last_window_width == expected_base_width
    assert app._ui_last_window_height <= 2160 / 2.0 * 0.92
    assert app.page.window.width == app._ui_last_window_width
    assert app.page.window.height == app._ui_last_window_height
    assert app.page.window.min_width == (
        app._estimate_main_panel_width() + S(24) * 2
    ) * 0.5


def test_measured_left_content_shrinks_window_and_root_together():
    app = _make_gui()
    app.device_pixel_ratio = 2.0
    app.display_scale = 1.0
    app.effective_ui_scale = 1.0
    app._display_resolution_for_window = lambda: (3840, 2160)
    app._fit_window_to_content(update=False, resize_window=False)
    app._ui_settings_content_height = 420.0

    asyncio.run(app._fit_window_to_measured_content())

    expected_height = 420.0 + S(24) * 2 + S(42)
    assert app._ui_last_window_height == expected_height
    assert app.page.window.height == expected_height
    assert app._root_row.height == app._ui_base_root_height
    assert app._ui_scale_host.height == app._ui_base_root_height


def test_programmatic_resize_event_updates_root_to_actual_page_height():
    async def run_scenario():
        app = _make_gui()
        app._loop = asyncio.get_running_loop()
        app.display_scale = 0.5
        app.effective_ui_scale = 0.5
        app._fit_window_to_content(update=False, resize_window=False)
        app._ui_programmatic_resize = True
        original_scale = app.effective_ui_scale

        app._on_page_resize(SimpleNamespace(width=800.0, height=600.0))

        expected_root_height = (
            600.0 - (S(24) * app.effective_ui_scale) * 2
        ) / app.effective_ui_scale
        assert app._ui_base_root_height == expected_root_height
        assert app.effective_ui_scale == original_scale

    asyncio.run(run_scenario())


def test_non_native_layout_refit_does_not_allow_width_rescale():
    app = _make_gui()
    app.device_pixel_ratio = 2.0
    app.effective_ui_scale = 0.8
    app.viewport_scale = 0.8

    app._fit_window_to_content(update=False, resize_window=False)
    app._on_page_resize(SimpleNamespace(width=700.0, height=600.0))

    assert app.effective_ui_scale == 0.8
    assert app.viewport_scale == 0.8


def test_dpr_only_media_change_preserves_viewport_scale():
    async def run_scenario():
        app = _make_gui()
        app.display_scale = 1.0
        app.device_pixel_ratio = 2.0
        app.viewport_scale = 0.8
        app.effective_ui_scale = 0.8
        app._display_resolution_for_window = lambda: (3840, 2160)
        update_calls = []
        app.page.update = lambda: update_calls.append(True)

        await app._apply_media_change()

        assert app.display_scale == 1.0
        assert app.effective_ui_scale == 0.8
        assert app.viewport_scale == 0.8
        assert update_calls == [True]

    asyncio.run(run_scenario())


def test_media_change_never_resizes_native_window_or_visual_scale():
    async def run_scenario():
        app = _make_gui()
        app.device_pixel_ratio = 2.0
        app.display_scale = 0.5
        app.effective_ui_scale = 0.5
        app._display_resolution_for_window = lambda: (3840, 2160)
        update_calls = []
        app.page.update = lambda: update_calls.append(True)

        await app._apply_media_change()

        assert app.display_scale == 0.5
        assert app.effective_ui_scale == 0.5
        assert update_calls == [True]

    asyncio.run(run_scenario())


def test_startup_layout_refits_after_dpi_settles():
    async def run_scenario():
        app = _make_gui()
        app.page.media = SimpleNamespace(device_pixel_ratio=2.0)
        calls = []
        app.viewport_scale = 1.25
        app.effective_ui_scale = 0.75
        app._fit_window_to_content = lambda **kwargs: calls.append(kwargs)

        await app._stabilize_startup_layout(delays=(0,))

        assert app.device_pixel_ratio == 2.0
        assert app.viewport_scale == 1.25
        assert app.effective_ui_scale == 0.75
        assert calls == [
            {"update": True, "resize_window": True},
        ]

    asyncio.run(run_scenario())
