import logging
import sys
import types


DSHOW_OUTPUT = """
[dshow @ 000001] "screen-capture-recorder" (video)
[dshow @ 000001] "Stereo Mix (Realtek(R) Audio)" (audio)
[dshow @ 000001] "virtual-audio-capturer" (audio)
[dshow @ 000001]   Alternative name "@device_cm_..."
"""

NEW_FFMPEG_DSHOW_OUTPUT = """
[in#0 @ 000001] "screen-capture-recorder" (video)
[in#0 @ 000001] "virtual-audio-capturer" (audio)
[in#0 @ 000001]   Alternative name "@device_sw_..."
"""


def _target():
    from gui.handlers import GUIHandlerMixin

    return object.__new__(GUIHandlerMixin)


def test_parse_ffmpeg_dshow_audio_devices() -> None:
    from streaming.audio import parse_ffmpeg_dshow_audio_devices

    assert parse_ffmpeg_dshow_audio_devices(DSHOW_OUTPUT) == [
        "Stereo Mix (Realtek(R) Audio)",
        "virtual-audio-capturer",
    ]


def test_parse_new_ffmpeg_dshow_audio_devices() -> None:
    from streaming.audio import parse_ffmpeg_dshow_audio_devices

    assert parse_ffmpeg_dshow_audio_devices(NEW_FFMPEG_DSHOW_OUTPUT) == [
        "virtual-audio-capturer",
    ]


def test_query_ffmpeg_dshow_audio_devices(monkeypatch, tmp_path) -> None:
    from streaming import audio

    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"")
    result = types.SimpleNamespace(stdout="", stderr=DSHOW_OUTPUT)
    monkeypatch.setattr(audio.subprocess, "run", lambda *args, **kwargs: result)

    assert audio.query_ffmpeg_dshow_audio_devices(ffmpeg) == [
        "Stereo Mix (Realtek(R) Audio)",
        "virtual-audio-capturer",
    ]


def test_windows_detection_prefers_dshow_virtual_audio(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: ["virtual-audio-capturer"],
    )
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: (_ for _ in ()).throw(
            AssertionError("sounddevice fallback must not run")
        )
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    target = _target()
    target._populate_audio_generic()

    assert target.audio_devices == ["virtual-audio-capturer"]
    assert not caplog.messages


def test_windows_detection_warns_when_dshow_has_no_loopback(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: ["Microphone (USB Audio)"],
    )

    target = _target()
    target._populate_audio_generic()

    assert target.audio_devices == ["virtual-audio-capturer"]
    assert caplog.messages == [
        "No Stereo Mix devices found, please enable it in audio settings.",
        "If no Stereo Mix, install 'Screen Capture Recorder':",
        (
            "https://github.com/rdp/"
            "screen-capture-recorder-to-video-windows-free/releases"
        ),
    ]


def test_windows_detection_falls_back_to_sounddevice(
    monkeypatch, caplog
) -> None:
    from gui import handlers

    caplog.set_level(logging.WARNING, logger="gui.handlers")
    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    monkeypatch.setattr(
        handlers,
        "query_ffmpeg_dshow_audio_devices",
        lambda path: None,
    )
    fake_sounddevice = types.SimpleNamespace(
        query_devices=lambda: [
            {
                "name": "Stereo Mix (Realtek(R) Audio)",
                "max_input_channels": 2,
                "max_output_channels": 0,
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    target = _target()
    target._populate_audio_generic()

    assert target.audio_devices == ["Stereo Mix (Realtek(R) Audio)"]
    assert not caplog.messages


def test_soundcard_sender_resolves_speaker_to_loopback_microphone(monkeypatch) -> None:
    from streaming.wasapi_audio import SoundcardLoopbackSender

    speaker = types.SimpleNamespace(id="speaker-id", name="Default speakers")
    loopback = types.SimpleNamespace(isloopback=True, name="Default speakers loopback")
    fake_soundcard = types.SimpleNamespace(
        all_speakers=lambda: [speaker],
        default_speaker=lambda: speaker,
        get_microphone=lambda *, id, include_loopback: (
            loopback if id == speaker.id and include_loopback else None
        ),
    )
    monkeypatch.setitem(sys.modules, "soundcard", fake_soundcard)

    sender = SoundcardLoopbackSender("virtual-audio-capturer")
    try:
        assert sender._loopback is loopback
    finally:
        sender.close()
