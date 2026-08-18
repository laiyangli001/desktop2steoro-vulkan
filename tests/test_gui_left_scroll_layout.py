from pathlib import Path

from gui.config import DEFAULTS


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = ROOT / "src" / "gui" / "builders.py"


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
    assert 'options=[v for v in aa_options], value="0"' in source
