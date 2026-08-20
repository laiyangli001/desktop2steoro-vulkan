from __future__ import annotations

import socket
import threading
import time
from typing import Any

import numpy as np


class SoundcardLoopbackSender:
    """Capture the Windows default speaker loopback and send PCM over localhost."""

    def __init__(self, device_name: str | None = None, *, samplerate: int = 48000):
        import soundcard as sc

        self.samplerate = int(samplerate)
        self.channels = 2
        self._soundcard = sc
        self._loopback = self._resolve_loopback(device_name)
        reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reservation.bind(("127.0.0.1", 0))
        self.port = int(reservation.getsockname()[1])
        reservation.close()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._startup_done = threading.Event()
        self._startup_error: Exception | None = None
        self._runtime_error: Exception | None = None
        self._thread: threading.Thread | None = None

    def _resolve_loopback(self, device_name: str | None) -> Any:
        requested = str(device_name or "").strip()
        speaker = None
        if requested:
            for candidate in self._soundcard.all_speakers():
                if str(getattr(candidate, "name", "")) == requested:
                    speaker = candidate
                    break
        if speaker is None:
            speaker = self._soundcard.default_speaker()
        if speaker is None:
            raise RuntimeError("No Windows default speaker is available")
        loopback = self._soundcard.get_microphone(
            id=speaker.id,
            include_loopback=True,
        )
        if loopback is None or not bool(getattr(loopback, "isloopback", False)):
            raise RuntimeError(
                f"No WASAPI loopback endpoint is available for speaker {speaker.name!r}"
            )
        return loopback

    @property
    def ffmpeg_url(self) -> str:
        return f"udp://127.0.0.1:{self.port}?fifo_size=65536&overrun_nonfatal=1"

    def start(self, *, timeout: float = 2.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="WasapiLoopback", daemon=True)
        self._thread.start()
        if not self._startup_done.wait(max(0.1, float(timeout))):
            self.close()
            raise RuntimeError("Windows loopback capture produced no PCM data")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError(
                f"Windows loopback capture failed: {type(error).__name__}: {error}"
            ) from error

    def _run(self) -> None:
        produced_pcm = False
        try:
            with self._loopback.recorder(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=1024,
            ) as recorder:
                while not self._stop.is_set():
                    samples = recorder.record(numframes=1024)
                    pcm = np.asarray(samples, dtype=np.float32)
                    pcm = np.clip(pcm, -1.0, 1.0)
                    self._socket.sendto(
                        (pcm * 32767.0).astype(np.int16).tobytes(),
                        ("127.0.0.1", self.port),
                    )
                    produced_pcm = True
                    self._startup_done.set()
        except Exception as exc:
            if self._stop.is_set():
                return
            if not produced_pcm:
                self._startup_error = exc
                self._startup_done.set()
                self._stop.set()
                return
            self._runtime_error = exc
            print(
                "[DirectSbsStream] Windows loopback capture interrupted: "
                f"{type(exc).__name__}: {exc}; continuing with silent audio",
                flush=True,
            )
            self._send_silence_until_stopped()

    def _send_silence_until_stopped(self) -> None:
        """Keep FFmpeg's mapped audio input alive after a capture interruption."""
        frames_per_packet = 1024
        packet = np.zeros(
            frames_per_packet * self.channels,
            dtype=np.int16,
        ).tobytes()
        interval = frames_per_packet / float(self.samplerate)
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                self._socket.sendto(packet, ("127.0.0.1", self.port))
            except OSError:
                if not self._stop.is_set():
                    raise
                return
            deadline += interval
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._socket.close()
