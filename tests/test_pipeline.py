import queue
import threading
from types import SimpleNamespace
from pathlib import Path

from stereo_runtime.pipeline import (
    _ParallelDepthScheduler,
    RuntimePipelineLoop,
    _enable_openxr_depth_cuda_graph_if_needed,
    _motion_sample,
    _motion_score,
    _runtime_motion_gate_enabled,
    _runtime_pending_depth_limit,
    _runtime_parallel_adaptive_backoff_enabled,
)


def test_parallel_depth_scheduler_enforces_effective_submission_limit():
    scheduler = object.__new__(_ParallelDepthScheduler)
    scheduler.worker_count = 3
    scheduler.effective_limit = 3
    scheduler.pending = [object(), object()]
    scheduler.trim = lambda _limit: None

    assert scheduler.can_submit() is True
    assert scheduler.set_effective_limit(2) is True
    assert scheduler.can_submit() is False
    assert scheduler.set_effective_limit(1) is True
    assert scheduler.effective_limit == 1


def test_parallel_scheduler_backoff_and_recovery_supports_three_workers(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", "1")
    class Scheduler:
        worker_count = 3
        effective_limit = 3

        def set_effective_limit(self, limit):
            self.effective_limit = max(1, min(self.worker_count, limit))
            return True

    events = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        source_stat_inc=lambda name, **kwargs: events.append((name, kwargs)),
        breakdown_inc=lambda *args, **kwargs: None,
    )
    loop._parallel_depth_scheduler = Scheduler()
    loop._parallel_backoff_until = 0.0
    loop._parallel_recovery_after = 0.0
    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 10.0)
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_BACKOFF_S", "1.5")

    loop._parallel_reduce_for_pressure("test")
    assert loop._parallel_depth_scheduler.effective_limit == 2
    loop._parallel_reduce_for_pressure("test")
    assert loop._parallel_depth_scheduler.effective_limit == 1

    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 11.6)
    loop._parallel_recover_if_ready()
    assert loop._parallel_depth_scheduler.effective_limit == 2
    monkeypatch.setattr("stereo_runtime.pipeline.time.perf_counter", lambda: 13.2)
    loop._parallel_recover_if_ready()
    assert loop._parallel_depth_scheduler.effective_limit == 3
    assert [name for name, _kwargs in events] == [
        "runtime_parallel_backoff",
        "runtime_parallel_backoff",
        "runtime_parallel_recovery",
        "runtime_parallel_recovery",
    ]


def test_parallel_adaptive_backoff_can_be_disabled(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PARALLEL_ADAPTIVE_BACKOFF", "0")
    assert _runtime_parallel_adaptive_backoff_enabled() is False

    class Scheduler:
        worker_count = 2
        effective_limit = 2

    events = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        source_stat_inc=lambda name, **kwargs: events.append(name),
        breakdown_inc=lambda *args, **kwargs: None,
    )
    loop._parallel_depth_scheduler = Scheduler()
    loop._parallel_reduce_for_pressure("test")

    assert loop._parallel_depth_scheduler.effective_limit == 2
    assert events == []


def test_openxr_motion_gate_matches_legacy_default(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_MOTION_GATE", raising=False)

    assert _runtime_motion_gate_enabled(SimpleNamespace(run_mode="OpenXR")) is False


def test_motion_gate_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_MOTION_GATE", "1")

    assert _runtime_motion_gate_enabled(SimpleNamespace(run_mode="OpenXR")) is True


def test_motion_gate_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_MOTION_GATE", "0")

    assert _runtime_motion_gate_enabled(SimpleNamespace(run_mode="OpenXR")) is False


def test_motion_sample_is_snapshot_when_input_storage_is_reused():
    import torch

    frame = torch.zeros((1, 3, 36, 64), dtype=torch.float32)
    previous = _motion_sample(frame)
    frame.fill_(1.0)
    current = _motion_sample(frame)

    assert _motion_score(previous, current) == 1.0


def test_openxr_pipeline_keeps_deferred_vulkan_request_path():
    source = (Path(__file__).resolve().parents[1] / "src/stereo_runtime/pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "ctx.stereo_runtime.process_openxr_frame(" in source
    assert "openxr_result_from_stereo_result" not in source
    assert 'output_mode="full_synthesis_eyes"' in source


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


def test_pending_cuda_retains_latest_raw_frame(monkeypatch):
    raw_q = queue.Queue(maxsize=1)
    raw_q.put_nowait("latest-frame")
    stats = []
    breakdown = []
    sleeps = []
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = SimpleNamespace(
        raw_q=raw_q,
        time_sleep=1.0 / 60.0,
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
        breakdown_inc=lambda name, *args, **kwargs: breakdown.append(name),
    )
    monkeypatch.setattr(
        "stereo_runtime.pipeline.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    loop._retain_latest_raw_while_cuda_pending()

    assert raw_q.get_nowait() == "latest-frame"
    assert stats == ["runtime_pending_cuda_inflight"]
    assert breakdown == ["runtime_pending_cuda_inflight"]
    assert sleeps == [0.001]


def _dual_pending_context(*, backend="cuda_triton", slots=2, temporal=False, workers=2):
    runtime = SimpleNamespace(
        depth_provider=SimpleNamespace(pipeline_slot_count=slots),
        config=SimpleNamespace(profile_sync=False),
        stereo_config=SimpleNamespace(temporal=temporal, convergence=0.0),
        _resolved_stereo_compute_backend=backend,
    )
    return SimpleNamespace(
        run_mode="OpenXR",
        openxr_runtime_direct=False,
        stereo_active_preset="cinema",
        stereo_runtime=runtime,
        runtime_config=SimpleNamespace(parallel_inference=True, parallel_inference_workers=workers),
    )


def test_openxr_safe_dual_slot_defaults_to_two_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context()) == 2


def test_openxr_three_slot_defaults_to_three_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(slots=3, workers=3)) == 3


def test_openxr_dual_pending_can_be_forced_back_to_one(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", "1")

    assert _runtime_pending_depth_limit(_dual_pending_context()) == 1


def test_openxr_dual_pending_requires_explicit_experimental_override(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", "2")

    assert _runtime_pending_depth_limit(_dual_pending_context()) == 2


def test_vulkan_deferred_pipeline_remains_single_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(backend="vulkan")) == 1


def test_unisolated_depth_provider_remains_single_pending(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_PENDING_CUDA_DEPTH", raising=False)

    assert _runtime_pending_depth_limit(_dual_pending_context(slots=1)) == 1


def test_dual_pending_backs_off_when_presenter_has_not_consumed_output():
    runtime_q = queue.Queue(maxsize=1)
    runtime_q.put_nowait("previous")
    loop = object.__new__(RuntimePipelineLoop)
    loop.context = _dual_pending_context()
    loop.context.runtime_q = runtime_q
    loop.context.runtime_ready_event = None
    loop.context.queue_put_latest = lambda q, item: (q.get_nowait(), q.put_nowait(item))
    loop.context.breakdown_inc = lambda *args, **kwargs: None
    loop.context.breakdown_add_time = lambda *args, **kwargs: None
    loop.context.source_stat_inc = lambda *args, **kwargs: None
    loop._dual_pending_cooldown_until = 0.0
    loop._last_algorithm_output_time = 0.0

    loop._publish_runtime_item((object(), 1.0, 0.01, 0.02, None))

    assert loop._pending_depth_limit() == 1


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
