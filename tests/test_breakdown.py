from utils.breakdown import FPSBreakdown


def test_fps_breakdown_logs_once_after_initial_delay(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    start = breakdown.last_log
    breakdown.inc("capture", 10)

    breakdown.log(now=start + 14.99)
    assert "[FPSBreakdown]" not in capsys.readouterr().out

    breakdown.log(now=start + 15.0)
    first = capsys.readouterr().out
    assert first.count("[FPSBreakdown]") == 1

    breakdown.log(now=start + 30.0)
    assert "[FPSBreakdown]" not in capsys.readouterr().out
