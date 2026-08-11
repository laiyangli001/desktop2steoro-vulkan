from pathlib import Path

from capture.adaptive_rate import AdaptiveCaptureRate
from gui.localization import get_messages


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = ROOT / "src" / "gui" / "builders.py"
HANDLERS_SOURCE = ROOT / "src" / "gui" / "handlers.py"


def test_auto_capture_steps_down_after_60_second_low_sbs_average() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    assert rate.observe_sbs_fps(20.0, now=1.0) == 60
    assert rate.observe_sbs_fps(20.0, now=30.0) == 60
    assert rate.observe_sbs_fps(20.0, now=61.0) == 25


def test_auto_capture_keeps_five_fps_headroom() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(27.0, now=1.0)
    rate.observe_sbs_fps(27.0, now=30.0)
    assert rate.observe_sbs_fps(27.0, now=61.0) == 32


def test_auto_capture_tracks_sbs_with_five_fps_headroom() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)
    for now in (1.0, 30.0, 61.0):
        rate.observe_sbs_fps(20.0, now=now)
    assert rate.current_fps() == 25

    assert rate.observe_sbs_fps(24.0, now=62.0) == 25
    assert rate.observe_sbs_fps(24.0, now=121.0) == 29
    assert rate.observe_sbs_fps(30.0, now=122.0) == 29
    assert rate.observe_sbs_fps(30.0, now=181.0) == 35


def test_auto_capture_does_not_jump_to_display_refresh_rate() -> None:
    rate = AdaptiveCaptureRate(120, enabled=True)

    rate.observe_sbs_fps(60.0, now=1.0)
    assert rate.observe_sbs_fps(60.0, now=61.0) == 65


def test_manual_capture_rate_is_not_adapted() -> None:
    rate = AdaptiveCaptureRate(60, enabled=False)

    for now in range(10):
        assert rate.observe_sbs_fps(10.0, now=float(now)) == 60


def test_capture_fps_gui_exposes_24_and_30_with_adaptive_tooltip() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    assert 'options=["Auto"] + [str(fps) for fps in range(5, 95, 5)]' in source
    handlers_source = HANDLERS_SOURCE.read_text(encoding="utf-8")
    assert '[t["Auto"]] + [str(fps) for fps in range(5, 95, 5)]' in handlers_source
    tooltip = get_messages("ZH")["tooltip_target_fps"]
    assert "SBS" in tooltip
    assert "+ 5 FPS" in tooltip
    return
    assert "每 60 秒" in tooltip
    assert "最多调整一个档位" in tooltip
