from __future__ import annotations

import sys

sys.path.insert(0, "src/desktop2stereo")

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
    assert GUI2_MESSAGES["CN"]["theme_system"] == "主题"


def test_community_defaults_do_not_invent_qq_material() -> None:
    assert WEBSITE_URL == "https://d2s.site"
    assert QQ_GROUP_NUMBER is None
    assert QQ_INVITE_URL is None
    assert qr_asset_path() is None
