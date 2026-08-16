from pathlib import Path


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
