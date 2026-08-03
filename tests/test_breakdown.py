from utils.breakdown import FPSBreakdown


def test_fps_breakdown_logs_five_times_at_15_second_intervals(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.inc("capture", 10)

    breakdown.log(now=start + 14.99)
    assert "[FPSBreakdown]" not in capsys.readouterr().out

    breakdown.log(now=start + 15.0)
    first = capsys.readouterr().out
    assert first.count("[FPSBreakdown]") == 1

    for index in range(2, 6):
        breakdown.log(now=start + 15.0 * index)
        assert capsys.readouterr().out.count("[FPSBreakdown]") == 1

    breakdown.log(now=start + 90.0)
    assert "[FPSBreakdown]" not in capsys.readouterr().out


def test_fps_breakdown_reports_normalized_hole_fill_mode(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.set_latest("rt_backend", "quality_4k")
    breakdown.set_latest("rt_hole_fill_mode", "quality")
    breakdown.set_latest("rt_hole_fill_radius", 3)
    breakdown.set_latest("rt_hole_fill_strength", 1.0)

    breakdown.log(now=start + 15.0)

    output = capsys.readouterr().out
    assert "rt_backend=quality_4k" in output
    assert "rt_hole_fill=quality(3/1.00)" in output
