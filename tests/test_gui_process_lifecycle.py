import asyncio

import gui.process as gui_process


def test_windows_process_tree_is_killed_before_parent(monkeypatch):
    events = []

    class FakeTreeKill:
        async def wait(self):
            events.append("taskkill.wait")
            return 0

    class FakeProcess:
        returncode = None

        def kill(self):
            events.append("parent.kill")

    async def fake_create_subprocess_exec(*args, **kwargs):
        events.append(tuple(args))
        return FakeTreeKill()

    monkeypatch.setattr(gui_process, "OS_NAME", "Windows")
    monkeypatch.setattr(
        gui_process.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    asyncio.run(
        gui_process.GUIProcessMixin._kill_process_tree(
            object(), FakeProcess(), 4242
        )
    )

    assert events == [
        ("taskkill", "/f", "/t", "/pid", "4242"),
        "taskkill.wait",
        "parent.kill",
    ]


def test_mediamtx_warning_is_not_promoted_by_error_text(monkeypatch):
    levels = []
    monkeypatch.setattr(
        gui_process.child_logger, "warning", lambda message: levels.append("warning")
    )
    monkeypatch.setattr(
        gui_process.child_logger, "error", lambda message: levels.append("error")
    )

    gui_process.GUIProcessMixin._log_child_line(
        object(),
        "[MediaMTX] 2026/08/19 19:05:54 WAR [HLS] segment changed - "
        "this will cause an error in iOS clients",
    )

    assert levels == ["warning"]


def test_graceful_stop_timeout_allows_runtime_cleanup():
    assert gui_process._GRACEFUL_PROCESS_STOP_TIMEOUT_S >= 8.0
