"""Cross-platform single-instance guard for the protected launcher."""

from __future__ import annotations

import atexit
import os
import platform
from pathlib import Path


class InstanceAlreadyRunning(RuntimeError):
    pass


class InstanceLock:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path or Path.home() / ".desktop2stereo" / "instance.lock")
        self._file = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b")
        try:
            if platform.system() == "Windows":
                import msvcrt

                self._file.seek(0)
                self._file.write(b"0")
                self._file.flush()
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self._file.close()
            self._file = None
            raise InstanceAlreadyRunning("Desktop2Stereo 已经在运行") from exc
        atexit.register(self.release)

    def release(self) -> None:
        if self._file is None:
            return
        try:
            if platform.system() == "Windows":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._file.close()
            self._file = None
