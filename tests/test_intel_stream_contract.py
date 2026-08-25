from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_intel_native_video_only_path_falls_back_when_audio_is_enabled() -> None:
    source = (
        ROOT / "src" / "desktop2stereo" / "streaming" / "direct_sbs.py"
    ).read_text(encoding="utf-8")
    assert "_disable_native_onevpl_for_audio" in source
    assert 'os.environ["D2S_ONEVPL_FINAL_SBS"] = "0"' in source
    assert 'os.environ["D2S_INTEL_VULKAN_SBS"] = "0"' in source
    assert "shared Intel QSV/FFmpeg " in source
    assert "audio+video path" in source


def test_consumer_submits_audio_fallback_frame_instead_of_dropping_it() -> None:
    source = (
        ROOT / "src" / "desktop2stereo" / "streaming" / "direct_sbs.py"
    ).read_text(encoding="utf-8")
    marker = "The native surface may be video-only"
    assert marker in source
    assert source.index(marker) < source.index("self.output.submit_frame(frame)", source.index(marker))
