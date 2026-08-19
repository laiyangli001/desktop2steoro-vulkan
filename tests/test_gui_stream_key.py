from types import SimpleNamespace


def _target():
    status_messages = []
    update_calls = []
    target = SimpleNamespace(
        locale="EN",
        _config={},
        set_status=status_messages.append,
        update_stream_url=lambda: update_calls.append(True),
    )
    return target, status_messages, update_calls


def test_stream_key_live_is_accepted() -> None:
    from gui.handlers import GUIHandlerMixin

    target, status_messages, update_calls = _target()
    event = SimpleNamespace(control=SimpleNamespace(value="live"))

    GUIHandlerMixin._on_stream_key_change(target, event)

    assert target._config["Stream Key"] == "live"
    assert status_messages == []
    assert update_calls == [True]


def test_invalid_stream_key_sets_status() -> None:
    from gui.handlers import GUIHandlerMixin

    target, status_messages, update_calls = _target()
    event = SimpleNamespace(control=SimpleNamespace(value="bad/key"))

    GUIHandlerMixin._on_stream_key_change(target, event)

    assert target._config["Stream Key"] == "bad/key"
    assert status_messages
    assert update_calls == [True]
