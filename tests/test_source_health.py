from __future__ import annotations

import logging

from capture.source_health import SourceHealth, format_age, safe_qsize


class BrokenQueue:
    def qsize(self):
        raise RuntimeError("unavailable")


class QueueStub:
    def __init__(self, size):
        self.size = size

    def qsize(self):
        return self.size


def test_safe_qsize_and_format_age():
    assert safe_qsize(QueueStub(3)) == 3
    assert safe_qsize(BrokenQueue()) == -1
    assert format_age(-1.0) == "n/a"
    assert format_age(1.234) == "1.23s"


def test_source_health_logs_stats(caplog):
    health = SourceHealth(
        enabled=True,
        run_mode="OpenXR",
        raw_q=QueueStub(1),
        runtime_q=QueueStub(2),
        source_active=lambda: True,
        render_active=lambda: False,
        idle_active=lambda: True,
    )
    health.inc("capture_frames", last_capture_ts=8.0)
    health.inc("raw_put")
    health.set(last_runtime_ts=9.0, last_process_latency=0.004, last_runtime_latency=0.006)

    with caplog.at_level(logging.DEBUG):
        health.log(now=10.0, force=True)

    output = caplog.text
    assert "DEBUG" in output
    assert "[Main] Source health:" in output
    assert "cap=1 raw_put=1" in output
    assert "raw_age=2.00s runtime_age=1.00s" in output
    assert "raw_q=1 runtime_q=2" in output
    assert "resize_ms=4.0 runtime_ms=6.0" in output
    assert "source=True render=False idle=True" in output


def test_source_health_limits_log_output(caplog):
    health = SourceHealth(
        enabled=True,
        run_mode="OpenXR",
        raw_q=QueueStub(0),
        runtime_q=QueueStub(0),
        source_active=lambda: True,
        render_active=lambda: False,
        idle_active=lambda: False,
    )

    with caplog.at_level(logging.DEBUG):
        for index in range(7):
            health.log(now=10.0 + index, force=True)

    assert caplog.text.count("[Main] Source health:") == 5


def test_source_health_skips_when_disabled_or_wrong_mode(caplog):
    disabled = SourceHealth(
        enabled=False,
        run_mode="OpenXR",
        raw_q=QueueStub(0),
        runtime_q=QueueStub(0),
        source_active=lambda: False,
        render_active=lambda: False,
        idle_active=lambda: False,
    )
    disabled.log(now=10.0, force=True)

    non_openxr = SourceHealth(
        enabled=True,
        run_mode="Viewer",
        raw_q=QueueStub(0),
        runtime_q=QueueStub(0),
        source_active=lambda: False,
        render_active=lambda: False,
        idle_active=lambda: False,
    )
    non_openxr.log(now=10.0, force=True)

    assert "Source health" not in caplog.text
