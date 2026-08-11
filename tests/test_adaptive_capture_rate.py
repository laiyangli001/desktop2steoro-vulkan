from pathlib import Path

from capture.adaptive_rate import AdaptiveCaptureRate
from gui.localization import get_messages


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = ROOT / "src" / "gui" / "builders.py"


def test_auto_capture_steps_down_after_60_second_low_sbs_average() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    assert rate.observe_sbs_fps(20.0, now=1.0) == 60
    assert rate.observe_sbs_fps(20.0, now=30.0) == 60
    assert rate.observe_sbs_fps(20.0, now=61.0) == 24


def test_auto_capture_uses_30_fps_bucket() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)

    rate.observe_sbs_fps(27.0, now=1.0)
    rate.observe_sbs_fps(27.0, now=30.0)
    assert rate.observe_sbs_fps(27.0, now=61.0) == 30


def test_reduced_capture_can_recover_one_step_per_60_second_window() -> None:
    rate = AdaptiveCaptureRate(60, enabled=True)
    for now in (1.0, 30.0, 61.0):
        rate.observe_sbs_fps(20.0, now=now)
    assert rate.current_fps() == 24

    assert rate.observe_sbs_fps(24.0, now=62.0) == 24
    assert rate.observe_sbs_fps(24.0, now=121.0) == 30
    assert rate.observe_sbs_fps(30.0, now=122.0) == 30
    assert rate.observe_sbs_fps(30.0, now=181.0) == 60


def test_manual_capture_rate_is_not_adapted() -> None:
    rate = AdaptiveCaptureRate(60, enabled=False)

    for now in range(10):
        assert rate.observe_sbs_fps(10.0, now=float(now)) == 60


def test_capture_fps_gui_exposes_24_and_30_with_adaptive_tooltip() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    assert '["Auto", "24", "30", "60", "72", "80", "90", "120"]' in source
    tooltip = get_messages("ZH")["tooltip_target_fps"]
    assert "每 60 秒" in tooltip
    assert "最多调整一个档位" in tooltip
