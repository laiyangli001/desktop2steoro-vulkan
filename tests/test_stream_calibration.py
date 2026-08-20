import json
import socket
import urllib.request

from streaming.stream_calibration import (
    CalibrationTier,
    StreamCalibrationController,
    calibration_tiers,
    evaluate_calibration_window,
    build_calibration_fingerprint,
)


def _receiver_report(fps=30.0, **overrides):
    report = {
        "decoded_fps": fps,
        "dropped_frames": 0,
        "freeze_count": 0,
        "packets_lost": 0,
        "bitrate_mbps": 20.0,
        "jitter_buffer_ms": 20.0,
    }
    report.update(overrides)
    return report


def test_calibration_tiers_are_ascending_and_bounded():
    tiers = calibration_tiers(50)

    assert [tier.fps for tier in tiers] == [30, 40, 48, 50]
    assert all(tier.target_mbps <= tier.peak_mbps for tier in tiers)


def test_calibration_window_requires_sender_and_receiver_stability():
    tier = CalibrationTier(fps=40, target_mbps=40, peak_mbps=46)
    reports = [_receiver_report(39.5) for _ in range(8)]

    passed, metrics = evaluate_calibration_window(
        tier, reports, {"submitted_fps": 39.4}
    )
    assert passed
    assert metrics["decoded_fps"] == 39.5

    failed, _ = evaluate_calibration_window(
        tier,
        reports[:-1] + [_receiver_report(39.5, freeze_count=1)],
        {"submitted_fps": 39.4},
    )
    assert not failed


def test_controller_advances_then_rolls_back_to_highest_stable_tier(tmp_path):
    now = [0.0]
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=40,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
        stage_seconds=2.0,
        clock=lambda: now[0],
    )

    assert controller.take_pending_tier().fps == 30
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(29.5))
    now[0] = 2.1
    controller.add_receiver_report(_receiver_report(29.5))

    assert controller.take_pending_tier().fps == 40
    controller.observe_sender({"submitted_fps": 32.0})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(31.0))
    now[0] = 4.2
    controller.add_receiver_report(_receiver_report(31.0))

    rollback = controller.take_pending_tier()
    assert rollback is not None and rollback.fps == 30
    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert profile["fps"] == 30
    assert controller.state()["status"] == "complete"


def test_controller_retries_stable_sender_with_lower_bitrate(tmp_path):
    now = [0.0]
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
        stage_seconds=2.0,
        clock=lambda: now[0],
    )
    original = controller.take_pending_tier()
    controller.observe_sender({"submitted_fps": 29.8})
    for _ in range(5):
        controller.add_receiver_report(_receiver_report(20.0, dropped_frames=2))
    now[0] = 2.1
    controller.add_receiver_report(_receiver_report(20.0, dropped_frames=2))

    retry = controller.take_pending_tier()
    assert retry is not None
    assert retry.fps == original.fps
    assert retry.target_mbps < original.target_mbps
    assert controller.state()["status"] == "reconnecting"


def test_headset_page_uses_whep_and_reports_browser_stats(tmp_path):
    controller = StreamCalibrationController(
        bind_port=12000,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
    )

    page = controller._viewer_html()
    assert "/whep" in page
    assert "getStats()" in page
    assert "framesDecoded" in page
    assert "jitterBufferDelay" in page


def test_calibration_fingerprint_tracks_runtime_and_receiver_choices():
    fingerprint = build_calibration_fingerprint(
        {
            "Computing Device": 0,
            "XR Headset Model": "Pico 4 / 4 Ultra",
            "Display Mode": "Half-SBS",
            "Depth Model": "Distill-Any-Depth-Base",
            "Stream Protocol": "WebRTC",
        }
    )

    assert fingerprint["Computing Device"] == "0"
    assert fingerprint["XR Headset Model"] == "Pico 4 / 4 Ultra"
    assert fingerprint["Stream Protocol"] == "WebRTC"


def test_feedback_http_server_serves_page_and_accepts_stats(tmp_path):
    reservation = socket.socket()
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]
    reservation.close()
    controller = StreamCalibrationController(
        bind_port=port,
        stream_port=1122,
        stream_key="live",
        maximum_fps=30,
        state_path=tmp_path / "state.json",
        profile_path=tmp_path / "profile.json",
    )
    controller.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
            assert "RTCPeerConnection" in response.read().decode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/stats",
            data=json.dumps(_receiver_report()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert json.loads(response.read())["ok"] is True
        assert controller.state()["receiver_connected"] is True
    finally:
        controller.close()
