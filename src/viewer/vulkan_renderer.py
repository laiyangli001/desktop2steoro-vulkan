from __future__ import annotations

from typing import Protocol


class VulkanWindowRenderer(Protocol):
    def initialize(self) -> None: ...

    def present(self, image_handle: object, ready_value: int) -> None: ...

    def close(self) -> None: ...

