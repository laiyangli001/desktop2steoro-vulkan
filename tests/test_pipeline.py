from types import SimpleNamespace

from stereo_runtime.pipeline import RuntimePipelineLoop


def test_pipeline_rebuilds_provider_after_consecutive_failures(monkeypatch):
    calls = []

    class Runtime:
        def _rebuild_depth_provider(self):
            calls.append("rebuild")

        def reset_temporal(self):
            calls.append("reset")

    stats = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        stereo_runtime=Runtime(),
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
    )
    loop._consecutive_runtime_errors = 3
    monkeypatch.setenv("D2S_RUNTIME_REBUILD_AFTER_ERRORS", "3")

    loop._rebuild_after_consecutive_failures()

    assert calls == ["rebuild", "reset"]
    assert "runtime_adapter_rebuilds" in stats
    assert loop._consecutive_runtime_errors == 0


def test_pipeline_does_not_rebuild_before_threshold(monkeypatch):
    calls = []

    class Runtime:
        def _rebuild_depth_provider(self):
            calls.append("rebuild")

    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        stereo_runtime=Runtime(),
        source_stat_inc=lambda *args, **kwargs: None,
    )
    loop._consecutive_runtime_errors = 2
    monkeypatch.setenv("D2S_RUNTIME_REBUILD_AFTER_ERRORS", "3")

    loop._rebuild_after_consecutive_failures()

    assert calls == []
    assert loop._consecutive_runtime_errors == 2
