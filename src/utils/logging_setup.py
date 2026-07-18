from __future__ import annotations

import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "desktop2stereo.log"


class _NoisyThirdPartyDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.name in ("flet", "flet_desktop", "flet_controls", "flet_transport") and record.levelno <= logging.DEBUG:
            return False
        if record.name == "PIL.PngImagePlugin" and message.startswith("STREAM "):
            return False
        return True


def configure_debug_file_logging(log_file: str | Path = LOG_FILE) -> None:
    """Write all logging output to the shared file without adding console noise."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    target = str(path.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == Path(target):
            handler.setLevel(logging.DEBUG)
            root.setLevel(logging.DEBUG)
            return

    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.addFilter(_NoisyThirdPartyDebugFilter())
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
