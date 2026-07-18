from __future__ import annotations

from viewer import window_utils


def test_is_window_visible_on_screen_matches_partial(monkeypatch):
    monkeypatch.setattr(
        window_utils,
        "list_windows",
        lambda: [("Stereo Viewer - Main", 1), ("Other", 2)],
    )

    assert window_utils.is_window_visible_on_screen("Stereo Viewer", timeout=0.01)


def test_is_window_visible_on_screen_matches_exact(monkeypatch):
    monkeypatch.setattr(
        window_utils,
        "list_windows",
        lambda: [("Stereo Viewer", 1)],
    )

    assert window_utils.is_window_visible_on_screen(
        "Stereo Viewer",
        partial_match=False,
        timeout=0.01,
    )


def test_is_window_visible_on_screen_returns_false_on_timeout(monkeypatch):
    monkeypatch.setattr(window_utils, "list_windows", lambda: [])
    monkeypatch.setattr(window_utils.time, "sleep", lambda _seconds: None)

    assert not window_utils.is_window_visible_on_screen("Missing", timeout=0.0)
