from __future__ import annotations

import os
import threading
from typing import Any

_RED = "\033[91m"
_RESET = "\033[0m"
_LOCK = threading.Lock()
_SEEN: set[str] = set()


def _ansi_enabled() -> bool:
    return os.environ.get("D2S_CPU_WARNING_COLOR", "1").strip().lower() not in {"0", "false", "no", "off"}


def _format_detail(detail: str | None) -> str:
    if not detail:
        return ""
    return f" detail={detail}"


def _emit(prefix: str, message: str) -> None:
    text = f"{prefix} {message}"
    if _ansi_enabled():
        text = f"{_RED}{text}{_RESET}"
    print(text, flush=True)


def red_cpu_warning(prefix: str, message: str, *, key: str | None = None, once: bool = True) -> None:
    if once:
        dedupe_key = key or f"{prefix}:{message}"
        with _LOCK:
            if dedupe_key in _SEEN:
                return
            _SEEN.add(dedupe_key)
    _emit(prefix, message)


def warn_cpu_fallback(component: str, reason: str, *, detail: str | None = None, key: str | None = None, once: bool = True) -> None:
    red_cpu_warning(
        "[CPU-FALLBACK]",
        f"{component}: reason={reason}{_format_detail(detail)}",
        key=key or f"fallback:{component}:{reason}:{detail or ''}",
        once=once,
    )


def warn_cpu_transfer(component: str, action: str, *, detail: str | None = None, key: str | None = None, once: bool = True) -> None:
    red_cpu_warning(
        "[CPU-TRANSFER]",
        f"{component}: action={action}{_format_detail(detail)}",
        key=key or f"transfer:{component}:{action}:{detail or ''}",
        once=once,
    )


def warn_cpu_operation(component: str, action: str, *, detail: str | None = None, key: str | None = None, once: bool = True) -> None:
    red_cpu_warning(
        "[CPU-OP]",
        f"{component}: action={action}{_format_detail(detail)}",
        key=key or f"operation:{component}:{action}:{detail or ''}",
        once=once,
    )


def describe_tensor(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    parts = []
    if shape is not None:
        try:
            parts.append(f"shape={tuple(shape)}")
        except Exception:
            parts.append(f"shape={shape}")
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    if device is not None:
        parts.append(f"device={device}")
    return " ".join(parts) if parts else "unknown"
