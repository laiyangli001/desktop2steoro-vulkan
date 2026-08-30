"""GUI2 community and external-link configuration.

Real QQ group material is deliberately left unset until the project owner
provides the verified group number, invite URL, and QR image.
"""

from __future__ import annotations

from pathlib import Path

from gui.paths import BASE_DIR


WEBSITE_URL = "https://d2s.site"
QQ_GROUP_NUMBER: str | None = None
QQ_INVITE_URL: str | None = None
QQ_QR_ASSET: str | None = None
COMMUNITY_ASSET_DIR = Path(BASE_DIR) / "gui2" / "assets" / "community"


def qr_asset_path() -> Path | None:
    if not QQ_QR_ASSET:
        return None
    path = COMMUNITY_ASSET_DIR / QQ_QR_ASSET
    return path if path.is_file() else None
