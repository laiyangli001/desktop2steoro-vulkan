from __future__ import annotations

from path_config import APP_ROOT


def test_vulkan_rtsp_soak_keeps_compressed_packet_boundary() -> None:
    source = (
        APP_ROOT.parent
        / "tools"
        / "vulkan_ffmpeg_rtsp_soak.py"
    ).read_text(encoding="utf-8")

    assert '"-f",\n                "h264"' in source
    assert '"-c:v",\n                "copy"' in source
    assert '"-rtsp_transport",\n                "tcp"' in source
    assert '"-pkt_size",\n                "1452"' in source
    assert "write(packet)" in source
    assert "duration-seconds" in source
    assert "gpu_to_cpu" not in source
    assert "rawvideo" not in source
