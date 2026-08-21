import asyncio
from pathlib import Path

from path_config import APP_ROOT

from gui.config import DEFAULTS
from gui.controls import CompactDisplayField, S
from gui.builders import GUIBuilderMixin


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"
HANDLERS_SOURCE = APP_ROOT / "gui" / "handlers.py"
GUI_SOURCE = APP_ROOT / "gui" / "gui.py"


def test_left_settings_area_uses_a_bounded_scroll_viewport() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    scroll_start = source.index("scroll_area = ft.Column([")
    scroll_end = source.index("self.log_level_dd", scroll_start)
    scroll_source = source[scroll_start:scroll_end]

    assert "scroll=ft.ScrollMode.AUTO" in scroll_source
    assert "expand=True" in scroll_source


def test_native_window_uses_a_visible_compact_startup_surface() -> None:
    source = GUI_SOURCE.read_text(encoding="utf-8")
    async_main_start = source.index("async def _async_main(page: ft.Page):")
    app_start = source.index("app = Desktop2StereoGUI(page)", async_main_start)
    bootstrap = source[async_main_start:app_start]

    assert "page.window.visible = False" not in bootstrap
    assert "page.window.width = S(520)" in bootstrap
    assert "page.window.height = S(300)" in bootstrap
    assert "ft.ProgressRing" in bootstrap
    assert "Desktop2Stereo is starting..." in bootstrap
    assert "page.update()" in bootstrap
    assert "await asyncio.sleep(0.1)" in bootstrap
    assert bootstrap.index("await asyncio.sleep(0.1)") < bootstrap.index(
        "_write_gui_ready_flag()"
    )


def test_startup_audio_detection_is_deferred_until_after_window_show() -> None:
    source = GUI_SOURCE.read_text(encoding="utf-8")
    setup_start = source.index("async def setup(self):")
    setup_end = source.index("def _signal_gui_ready", setup_start)
    setup_source = source[setup_start:setup_end]

    show_index = setup_source.index("self.page.window.visible = True")
    audio_index = setup_source.index("self.populate_audio_devices_after_startup()")
    assert show_index < audio_index

    handlers = HANDLERS_SOURCE.read_text(encoding="utf-8")
    async_start = handlers.index("async def populate_audio_devices_after_startup")
    async_end = handlers.index("def _populate_audio_generic", async_start)
    assert "await asyncio.to_thread" in handlers[async_start:async_end]


def test_torch_device_detection_is_deferred_until_after_window_show() -> None:
    source = GUI_SOURCE.read_text(encoding="utf-8")
    setup_start = source.index("async def setup(self):")
    setup_end = source.index("def _signal_gui_ready", setup_start)
    setup_source = source[setup_start:setup_end]

    assert "self.populate_devices()" not in setup_source
    show_index = setup_source.index("self.page.window.visible = True")
    device_index = setup_source.index("self.populate_devices_after_startup()")
    assert show_index < device_index

    builders = BUILDERS_SOURCE.read_text(encoding="utf-8")
    build_start = builders.index("def build_ui(self):")
    build_end = builders.index("def _build_streamer_rows", build_start)
    assert "DEVICES.values()" not in builders[build_start:build_end]
    assert 'options=["Detecting compute devices..."]' in builders[build_start:build_end]
    assert "async def populate_devices_after_startup" in builders


def test_left_scroll_area_contains_the_action_footer() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")

    assert "scroll_area.controls.append(footer)" in source
    assert "content=scroll_area" in source


def test_stop_and_run_buttons_are_inset_from_the_right_edge() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    row_start = source.index("btn_row = ft.Row(")
    row_end = source.index("self._btn_bar =", row_start)

    row_source = source[row_start:row_end]
    assert "padding=ft.Padding(0, 0, S(20), 0)" in row_source
    assert "spacing=S(20)" in row_source


def test_stop_and_run_buttons_have_matching_widths() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    controls_start = source.index("self.stop_btn = ft.Button(")
    controls_end = source.index("lang_row =", controls_start)
    controls_source = source[controls_start:controls_end]

    assert controls_source.count("width=S(130)") == 2


def test_main_panels_stretch_to_the_page_height() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    root_start = source.index("self._root_row = ft.Row(")
    root_end = source.index("page.add(self._root_row)", root_start)
    root_source = source[root_start:root_end]

    assert "vertical_alignment=ft.CrossAxisAlignment.STRETCH" in root_source


def test_startup_fit_is_not_cancelled_by_an_early_resize_event(monkeypatch) -> None:
    class WindowHarness:
        def __init__(self):
            self.width = 0
            self.update_calls = 0

        def update(self):
            self.update_calls += 1

    class PageHarness:
        def __init__(self):
            self.window = WindowHarness()

    class StartupFitHarness(GUIBuilderMixin):
        def __init__(self):
            self._closed = False
            self._startup_fit_armed = True
            self.fit_calls = 0
            self.page = PageHarness()
            self.run_mode_key = "RTMP Streamer"
            self.log_panel = None

        def _estimate_main_panel_width(self):
            return 513

        def _estimate_window_width(self, main_width=None):
            return 568

        async def _resize_window_after_log_visibility_change(self):
            self.fit_calls += 1
            self.page.window.width = 568

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("gui.builders.asyncio.sleep", no_wait)
    gui = StartupFitHarness()

    asyncio.run(gui._apply_startup_fit())

    assert gui.fit_calls == 2
    assert gui.page.window.update_calls == 1
    assert gui.page.window.width == 568
    assert gui._startup_fit_armed is False


def test_stream_calibration_uses_the_shared_label_and_control_widths() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    align_start = source.index("left_labels = [")
    align_end = source.index("right_labels = [", align_start)
    row_start = source.index("self.stream_calibration_mode_dd = CompactDropdown(")
    row_end = source.index("self.crf_label =", row_start)

    assert "self.stream_calibration_label" in source[align_start:align_end]
    assert source[row_start:row_end].count("width=S(130)") == 2
    row_controls_start = source.index("self.stream_calibration_row = ft.Row(")
    row_controls_end = source.index("self.crf_label =", row_controls_start)
    row_controls = source[row_controls_start:row_controls_end]
    assert row_controls.index("self.stream_calibration_mode_dd") < row_controls.index(
        "self.stream_calibration_btn"
    )
    between = row_controls[
        row_controls.index("self.stream_calibration_mode_dd"):
        row_controls.index("self.stream_calibration_btn")
    ]
    assert "ft.Container(width=S(10))" in between
    status_gap = row_controls[
        row_controls.index("self.stream_calibration_btn"):
        row_controls.index("self.stream_calibration_status")
    ]
    assert "ft.Container(width=S(10))" in status_gap


def test_stream_url_field_has_a_bounded_width() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    field_start = source.index("self.stream_url_tf = CompactDisplayField(")
    field_end = source.index("self.preview_btn =", field_start)
    field_source = source[field_start:field_end]

    assert "min_width=S(130)" in field_source
    assert "max_width=S(230)" in field_source


def test_compact_display_field_adapts_between_minimum_and_maximum_widths() -> None:
    field = CompactDisplayField("short", min_width=S(130), max_width=S(230))
    assert field.width == S(130)

    field.value = "https://example.test/" + "stream/" * 100
    assert field.width == S(230)
    estimator = object.__new__(GUIBuilderMixin)
    assert estimator._estimate_control_width(field) == S(230)


def test_window_height_reserves_the_complete_action_footer() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    estimator_start = source.index("def _estimate_window_height")
    estimator_end = source.index("# ── label alignment", estimator_start)
    estimator_source = source[estimator_start:estimator_end]

    assert "footer_height = S(84)" in estimator_source


def test_window_preview_is_an_independent_advanced_checkbox_after_vsync() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    row_start = source.index("self.row6b = ft.Row(")
    row_end = source.index("self.render_policy_label", row_start)
    row_source = source[row_start:row_end]

    assert "self.local_vsync_cb" in row_source
    assert "self.window_preview_cb" in row_source
    assert row_source.index("self.local_vsync_cb") < row_source.index("self.window_preview_cb")


def test_reset_defaults_disable_depth_antialiasing() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")

    assert DEFAULTS["Anti-aliasing"] == 0
    assert DEFAULTS["Depth Antialias Strength"] == 0.0
    assert DEFAULTS["CRF"] == 23
    assert DEFAULTS["Stream Protocol"] == "WebRTC"
    assert 'options=[v for v in aa_options], value="0"' in source


def test_run_mode_change_refits_native_window_height() -> None:
    source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("def on_run_mode_change")
    end = source.index("def on_advanced_device_change", start)
    handler = source[start:end]

    assert "self._fit_window_to_content(update=True, resize_window=True)" in handler


def test_reset_defaults_refits_window_to_left_gui_content() -> None:
    source = (APP_ROOT / "gui" / "process.py").read_text(encoding="utf-8")
    reset_start = source.index("def reset_defaults")
    reset_end = source.index("# ── URL actions ──", reset_start)
    reset_source = source[reset_start:reset_end]

    assert "self._fit_window_to_content(update=True, resize_window=True)" in reset_source


def test_headset_model_uses_a_dedicated_row_below_run_mode() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    run_row_start = source.index("self.row7a = ft.Row(")
    headset_row_start = source.index("self.xr_headset_row = ft.Row(")
    assembly_start = source.index("device_group = ft.Container(")
    assembly_end = source.index("lang_group = ft.Container(", assembly_start)
    assembly = source[assembly_start:assembly_end]

    assert "self.xr_headset_label" not in source[run_row_start:headset_row_start]
    assert assembly.index("self.row7a") < assembly.index("self.xr_headset_row")
    assert assembly.index("self.xr_headset_row") < assembly.index("self.row7b")


def test_headset_model_remains_visible_in_every_run_mode() -> None:
    source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("def _sync_visibility(self):")
    end = source.index("def ", start + 4)
    visibility = source[start:end]

    assert "self.xr_headset_label.visible = True" in visibility
    assert "self.xr_headset_dd.visible = True" in visibility


def test_refresh_button_follows_the_capture_source_dropdown() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    row_start = source.index("row8 = ft.Row(")
    row_end = source.index("# Row 10:", row_start)
    row_source = source[row_start:row_end]

    assert "ft.Container(expand=True)" not in row_source
    assert row_source.index("self.monitor_dd") < row_source.index("self.refresh_btn")
    assert row_source.index("self.window_dd") < row_source.index("self.refresh_btn")
