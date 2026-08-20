from __future__ import annotations

from typing import Protocol


class OpenGlFallbackRenderer(Protocol):
    def initialize(self) -> None: ...

    def present(self, image_handle: object) -> None: ...

    def close(self) -> None: ...

