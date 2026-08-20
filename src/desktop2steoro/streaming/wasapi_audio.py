from __future__ import annotations

import socket
import threading
from typing import Any

import numpy as np


class SoundcardLoopbackSender:
    """Capture the Windows default speaker loopback and send PCM over localhost."""

    def __init__(self, device_name: str | None = None, *, samplerate: int = 48000):
        import soundcard as sc

        self.samplerate = int(samplerate)
        self.channels = 2
        self._soundcard = sc
        self._speaker = self._resolve_speaker(device_name)
        reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reservation.bind(("127.0.0.1", 0))
        self.port = int(reservation.getsockname()[1])
        reservation.close()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._startup_done = threading.Event()
        self._startup_error: Exception | None = None
        self._thread: threading.Thread | None = None

    def _resolve_speaker(self, device_name: str | None) -> Any:
        requested = str(device_name or "").strip()
        if requested:
            for speaker in self._soundcard.all_speakers():
                if str(getattr(speaker, "name", "")) == requested:
                    return speaker
        speaker = self._soundcard.default_speaker()
        if speaker is None:
            raise RuntimeError("No Windows default speaker is available")
        return speaker

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
        try:
            with self._speaker.loopback().recorder(
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
                    self._startup_done.set()
        except Exception as exc:
            self._startup_error = exc
            self._startup_done.set()
            if not self._stop.is_set():
                self._stop.set()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._socket.close()
