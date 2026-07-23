import queue
import threading
from types import SimpleNamespace

from stereo_runtime.pipeline import RuntimePipelineLoop, _enable_openxr_depth_cuda_graph_if_needed


def test_publish_runtime_item_marks_inference_pipeline_ready():
    ready = threading.Event()
    runtime_q = queue.Queue(maxsize=1)
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        runtime_q=runtime_q,
        runtime_ready_event=ready,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        source_stat_inc=lambda *args, **kwargs: None,
    )

    loop._publish_runtime_item((object(), 1.0, 0.01, 0.02, None))

    assert ready.is_set()
    assert runtime_q.qsize() == 1


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


def test_cuda_capture_disables_previously_enabled_openxr_cuda_graph():
    snapshots = []

    class Runtime:
        config = SimpleNamespace(depth_backend="tensorrt_native", use_cuda_graph=True)

        def apply_settings_snapshot(self, snapshot, *, active_preset):
            snapshots.append(snapshot)
            self.config.use_cuda_graph = snapshot.use_cuda_graph

    stats = []
    ctx = SimpleNamespace(
        stereo_runtime=Runtime(),
        stereo_active_preset="quality_4k",
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
    )

    _enable_openxr_depth_cuda_graph_if_needed(
        ctx,
        True,
        SimpleNamespace(capture_tool="WindowsCaptureCUDA"),
    )

    assert [snapshot.use_cuda_graph for snapshot in snapshots] == [False]
    assert ctx.stereo_runtime.config.use_cuda_graph is False
    assert "openxr_depth_cuda_graph_disabled_cuda_capture" in stats
