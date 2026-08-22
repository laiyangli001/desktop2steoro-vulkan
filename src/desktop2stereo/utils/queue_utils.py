from __future__ import annotations

import queue
from typing import Any, Callable


def put_latest(q: queue.Queue, item: Any) -> None:
    """Keep only the newest item without blocking producer threads."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def clear_nonblocking(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def drain_latest(
    q: queue.Queue,
    first_item: Any,
    *,
    on_drop: Callable[[], None] | None = None,
) -> Any:
    """Drop stale queued items and return the newest available frame."""
    latest = first_item
    while True:
        try:
            latest = q.get_nowait()
            if on_drop is not None:
                on_drop()
        except queue.Empty:
            return latest
