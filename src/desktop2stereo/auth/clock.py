"""Persisted server-time checkpoint used to detect local clock rollback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


ROLLBACK_TOLERANCE_SECONDS = 300


class ClockSuspectError(RuntimeError):
    """The local clock moved too far behind the last trusted server time."""


class TrustedClock:
    def __init__(self, app_dir: str | os.PathLike | None = None):
        root = Path(app_dir or Path.home() / ".desktop2stereo")
        self.path = root / "trusted-clock.json"

    def observe(self, server_time: int, *, local_time: int | None = None) -> None:
        trusted = self._read()
        current = int(time.time() if local_time is None else local_time)
        if trusted is not None and current + ROLLBACK_TOLERANCE_SECONDS < trusted:
            raise ClockSuspectError("本机系统时间异常，请校准时间后重新联网验证授权")
        if server_time > (trusted or 0):
            self._write(server_time)

    def check(self, *, local_time: int | None = None) -> None:
        trusted = self._read()
        if trusted is None:
            return
        current = int(time.time() if local_time is None else local_time)
        if current + ROLLBACK_TOLERANCE_SECONDS < trusted:
            raise ClockSuspectError("本机系统时间异常，请校准时间后重新联网验证授权")

    def _read(self) -> int | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            timestamp = int(value["max_server_time"])
            return timestamp if timestamp > 0 else None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _write(self, timestamp: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"max_server_time": int(timestamp)}), encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, self.path)
