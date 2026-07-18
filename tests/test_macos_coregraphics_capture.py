from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "capture" / "backends" / "macos_coregraphics.py"


def _source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_coregraphics_window_rect_is_refreshed_before_capture():
    source = _source()

    assert "def _ensure_rect(self):" in source
    assert 'if self.capture_mode != "Monitor":' in source
    assert 'current = (info["left"], info["top"], info["width"], info["height"])' in source
    assert "self.window_id = info[\"window_id\"]" in source
    assert "self.left, self.top, self.width, self.height = current" in source

    grab_start = source.index("    def grab(self, output_format=\"bgr\"):")
    ensure_call = source.index("        self._ensure_rect()", grab_start)
    capture_call = source.index("            frame = _cg_capture_region_as_bgra", grab_start)

    assert ensure_call < capture_call


def test_coregraphics_cursor_position_uses_refreshed_rect_and_scale():
    source = _source()

    assert "if 0 <= x - self.left <= self.width and 0 <= y - self.top <= self.height:" in source
    assert "cursor_x = (x - self.left) * system_scale" in source
    assert "cursor_y = (y - self.top) * system_scale" in source
