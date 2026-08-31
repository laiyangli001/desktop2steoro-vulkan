from __future__ import annotations

import sys

sys.path.insert(0, "src/desktop2stereo")

from gui2 import community as community_module
from gui2.community import QQ_GROUP_NUMBER, QQ_INVITE_URL, WEBSITE_URL, qr_asset_path
from gui2.localization import GUI2_MESSAGES
from gui2.menu_registry import build_menu_specs


def test_gui2_menu_has_stable_top_level_structure() -> None:
    specs = build_menu_specs()
    assert [item.item_id for item in specs] == ["settings", "tools"]
    assert [item.item_id for item in specs[0].children] == ["reset_defaults"]


def test_gui2_menu_has_bilingual_labels_and_all_theme_choices() -> None:
    for locale in ("EN", "CN"):
        catalog = GUI2_MESSAGES[locale]
        assert catalog["menu_settings"]
        assert catalog["menu_help"]
        assert catalog["nav_logs"]
        assert catalog["nav_help"]
        assert catalog["help_title"]
        for theme in ("system", "blue", "green", "red", "purple", "orange", "teal", "pink", "grey"):
            assert catalog[f"theme_{theme}"]
        for page in (
            "home", "stereo", "quality", "performance",
            "streaming", "advanced", "logs", "help",
        ):
            description = catalog[f"{page}_description"]
            assert description
            assert description != catalog[f"{page}_title"]
            assert len(description) > len(catalog[f"{page}_title"])
    assert GUI2_MESSAGES["CN"]["theme_system"] == "主题"


def test_community_material_points_to_the_project_qq_group() -> None:
    assert WEBSITE_URL == "https://d2s.site"
    assert QQ_GROUP_NUMBER == "621378639"
    assert QQ_INVITE_URL is None
    assert qr_asset_path() is not None
    assert qr_asset_path().name == "d2s_qq.jpg"


def test_remote_qq_image_refresh_requires_a_help_visit_gap_over_seven_days(
    monkeypatch, tmp_path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "d2s_qq.jpg"
    visit_stamp = cache_dir / "d2s_qq.last_help_visit"
    download_calls = []

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b"first-image" if len(download_calls) == 1 else b"second-image"

    def fake_urlopen(_request, timeout):
        download_calls.append(timeout)
        return _Response()

    monkeypatch.setattr(community_module, "QQ_QR_CACHE_DIR", cache_dir)
    monkeypatch.setattr(community_module, "QQ_QR_CACHE_PATH", cache_path)
    monkeypatch.setattr(
        community_module, "QQ_QR_LAST_HELP_VISIT_STAMP", visit_stamp,
    )
    monkeypatch.setattr(community_module, "urlopen", fake_urlopen)

    first_source = community_module.qr_asset_source(now=1_000.0)
    assert first_source == cache_path
    assert cache_path.read_bytes() == b"first-image"
    assert len(download_calls) == 1

    exactly_seven_days = (
        1_000.0 + community_module.QQ_QR_HELP_VISIT_INTERVAL_SECONDS
    )
    cached_source = community_module.qr_asset_source(
        now=exactly_seven_days,
    )
    assert cached_source == cache_path
    assert cache_path.read_bytes() == b"first-image"
    assert len(download_calls) == 1

    refreshed_source = community_module.qr_asset_source(
        now=(
            exactly_seven_days
            + community_module.QQ_QR_HELP_VISIT_INTERVAL_SECONDS
            + 1
        ),
    )
    assert refreshed_source == cache_path
    assert cache_path.read_bytes() == b"second-image"
    assert len(download_calls) == 2
