import logging


def test_debug_file_logging_writes_debug_without_console_handler(tmp_path):
    from utils.logging_setup import configure_debug_file_logging

    log_file = tmp_path / "desktop2stereo.log"
    logger = logging.getLogger("d2s_test_debug_file")
    flet_logger = logging.getLogger("flet_controls")
    pil_logger = logging.getLogger("PIL.PngImagePlugin")
    transport_logger = logging.getLogger("flet_transport")
    root = logging.getLogger()
    before = list(root.handlers)

    configure_debug_file_logging(log_file)
    try:
        logger.debug("debug detail")
        flet_logger.debug("Container(1 - 2).did_mount()")
        flet_logger.debug("Text(1 - 2).will_unmount()")
        flet_logger.debug("Trigger event Page(1 - 2).on_resize PageResizeEvent(name='resize')")
        flet_logger.warning("important flet warning")
        pil_logger.debug("STREAM b'IHDR' 16 13")
        pil_logger.warning("important png warning")
        transport_logger.debug("send_message: ClientMessage(action=<ClientActions.UPDATE>)")
        transport_logger.debug("_on_message: ServerMessage(action=<ServerActions.PAGE_EVENT>)")
        transport_logger.warning("important transport warning")

        for handler in root.handlers:
            handler.flush()

        text = log_file.read_text(encoding="utf-8")
        assert "debug detail" in text
        assert "important flet warning" in text
        assert "important png warning" in text
        assert "important transport warning" in text
        assert "send_message:" not in text
        assert "_on_message:" not in text
        assert "did_mount" not in text
        assert "will_unmount" not in text
        assert "Trigger event" not in text
        assert "STREAM b'IHDR'" not in text
        assert len(root.handlers) == len(before) + 1
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_i18n_locale_completeness():
    """All locale keys must match EN keys."""
    from utils.i18n import MESSAGES

    en_keys = set(MESSAGES["EN"].keys())
    for loc in MESSAGES:
        loc_keys = set(MESSAGES[loc].keys())
        missing = en_keys - loc_keys
        extra = loc_keys - en_keys
        assert not missing, f"Locale {loc} missing keys: {missing}"
        assert not extra, f"Locale {loc} has extra keys not in EN: {extra}"


def test_i18n_t_function_fallback():
    from utils.i18n import t

    assert "nonexistent_key" == t("nonexistent_key")
    assert "nonexistent_key" == t("nonexistent_key", "CN")
    assert "📦 Preparing environment..." == t("Preparing environment", "EN")
    assert "📦 正在准备运行环境..." == t("Preparing environment", "CN")


def test_i18n_t_formatting():
    from utils.i18n import t

    result = t("Downloading model", "EN", model="DepthPro")
    assert "DepthPro" in result
    assert "⬇️" in result


def test_i18n_status_log_accepts_level():
    from utils.i18n import status_log
    import logging
    import os

    os.environ["DESKTOP2STEREO_LOCALE"] = "EN"
    status_log("Ready", level=logging.INFO)
    status_log("Fatal error", level=logging.ERROR, error="test error")

