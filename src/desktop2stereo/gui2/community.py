"""GUI2 community and external-link configuration."""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from gui.paths import BASE_DIR


WEBSITE_URL = "https://d2s.site"
QQ_GROUP_NUMBER: str | None = "621378639"
QQ_INVITE_URL: str | None = None
QQ_QR_ASSET: str | None = "d2s_qq.jpg"
QQ_QR_URL: str | None = "https://d2s.site/d2s_qq.jpg"
COMMUNITY_ASSET_DIR = Path(BASE_DIR) / "gui2" / "assets" / "community"
QQ_QR_HELP_VISIT_INTERVAL_SECONDS = 7 * 24 * 60 * 60
QQ_QR_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
QQ_QR_CACHE_DIR = Path(BASE_DIR) / "logs" / "gui2_community"
QQ_QR_CACHE_PATH = QQ_QR_CACHE_DIR / "d2s_qq.jpg"
QQ_QR_LAST_HELP_VISIT_STAMP = QQ_QR_CACHE_DIR / "d2s_qq.last_help_visit"


def qr_asset_path() -> Path | None:
    if not QQ_QR_ASSET:
        return None
    path = COMMUNITY_ASSET_DIR / QQ_QR_ASSET
    return path if path.is_file() else None


def _last_help_visit_time() -> float | None:
    try:
        return float(QQ_QR_LAST_HELP_VISIT_STAMP.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _download_due_for_help_visit(now: float) -> bool:
    last_visit = _last_help_visit_time()
    return (
        last_visit is None
        or now - last_visit > QQ_QR_HELP_VISIT_INTERVAL_SECONDS
    )


def _record_help_visit(now: float) -> None:
    try:
        QQ_QR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QQ_QR_LAST_HELP_VISIT_STAMP.write_text(str(now), encoding="utf-8")
    except OSError:
        pass


def _download_qr_to_cache() -> bool:
    if not QQ_QR_URL:
        return False
    temporary_path = QQ_QR_CACHE_PATH.with_suffix(".download")
    try:
        request = Request(QQ_QR_URL, method="GET")
        request.add_header("User-Agent", "Desktop2Stereo-GUI2/1.0")
        with urlopen(request, timeout=3.0) as response:
            status = int(getattr(response, "status", 200))
            if not 200 <= status < 400:
                return False
            payload = response.read(QQ_QR_MAX_DOWNLOAD_BYTES + 1)
        if not payload or len(payload) > QQ_QR_MAX_DOWNLOAD_BYTES:
            return False
        QQ_QR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(payload)
        os.replace(temporary_path, QQ_QR_CACHE_PATH)
        return True
    except (OSError, URLError, ValueError):
        return False
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def qr_asset_source(*, now: float | None = None) -> Path | None:
    """Refresh after a Help-page visit gap longer than seven days."""
    current_time = time.time() if now is None else float(now)
    download_due = _download_due_for_help_visit(current_time)
    # This function is called only when the Help page is explicitly opened.
    # Recording every visit makes the next decision depend on the interval
    # between Help-page visits, regardless of whether a download was needed.
    _record_help_visit(current_time)
    if QQ_QR_URL and download_due:
        _download_qr_to_cache()
    if QQ_QR_CACHE_PATH.is_file():
        return QQ_QR_CACHE_PATH
    return qr_asset_path()
