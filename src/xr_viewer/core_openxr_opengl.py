from __future__ import annotations

from typing import Protocol


class OpenXrOpenGlFallbackPresenter(Protocol):
    def initialize(self) -> None: ...

    def run_frame(self) -> bool: ...

    def close(self) -> None: ...

