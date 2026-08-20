from pathlib import Path

from gui.config import DEFAULTS


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = ROOT / "src" / "gui" / "builders.py"
HANDLERS_SOURCE = ROOT / "src" / "gui" / "handlers.py"


def test_left_settings_area_uses_a_bounded_scroll_viewport() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    scroll_start = source.index("scroll_area = ft.Column([")
    scroll_end = source.index("self.log_level_dd", scroll_start)
    scroll_source = source[scroll_start:scroll_end]

    assert "scroll=ft.ScrollMode.AUTO" in scroll_source
    assert "expand=True" in scroll_source


def test_left_scroll_area_contains_the_action_footer() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")

    assert "scroll_area.controls.append(footer)" in source
    assert "content=scroll_area" in source


def test_main_panels_stretch_to_the_page_height() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    root_start = source.index("self._root_row = ft.Row(")
    root_end = source.index("page.add(self._root_row)", root_start)
    root_source = source[root_start:root_end]

    assert "vertical_alignment=ft.CrossAxisAlignment.STRETCH" in root_source


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
    source = (ROOT / "src" / "gui" / "process.py").read_text(encoding="utf-8")
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
