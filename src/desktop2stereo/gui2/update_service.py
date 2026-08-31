"""Update-service boundary for GUI2.

The previous Windows batch updater targets the legacy ``lc700x/desktop2stereo``
project.  GUI2 deliberately keeps update operations disabled until a new
project endpoint and a safe replacement strategy are defined.
"""

from __future__ import annotations

from dataclasses import dataclass


UPDATE_FEATURE_ENABLED = False


@dataclass(frozen=True)
class UpdateCheckResult:
    """Result returned by the future update implementation."""

    available: bool
    message_key: str


class UpdateService:
    """Stable GUI-facing boundary for checking and applying updates."""

    enabled = UPDATE_FEATURE_ENABLED

    def check_for_updates(self) -> UpdateCheckResult:
        """Return a disabled result without network or process activity."""
        if not self.enabled:
            return UpdateCheckResult(False, "update_feature_disabled")
        raise NotImplementedError("The new-project update service is not configured")
