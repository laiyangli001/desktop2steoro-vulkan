from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from app_runtime.runtime_output import (
    CudaVulkanOutputAdapter,
    VulkanRuntimeOutputConsumer,
)


def test_screen_light_sample_completion_is_non_blocking_and_clamped():
    adapter = CudaVulkanOutputAdapter(None)
    adapter._screen_light_pending = (
        SimpleNamespace(tolist=lambda: [-1.0, 0.25, 12.0]),
        SimpleNamespace(query=lambda: True),
    )
    adapter._screen_light_last_submit = time.monotonic()

    adapter._update_screen_light_sample(object(), object())

    assert adapter._screen_light_rgb == (0.0, 0.25, 8.0)
    assert adapter._screen_light_pending is None


def test_consumer_rejects_non_vulkan_results_without_cpu_conversion():
    runtime_q = queue.Queue(maxsize=1)
    shutdown = threading.Event()
    stats = []
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append((name, amount, values)),
    )

    assert consumer._to_output_frame((SimpleNamespace(left_eye="cuda", right_eye="cuda"), 1.0)) is None
    assert any(item[0] == "runtime_output_waiting_for_vulkan_importer" for item in stats)


def test_consumer_overwrites_stale_queue_items():
    runtime_q = queue.Queue(maxsize=2)
    shutdown = threading.Event()
    stats = []
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append((name, amount, values)),
    )
    runtime_q.put((SimpleNamespace(left_eye="old", right_eye="old"), 1.0))
    runtime_q.put((SimpleNamespace(left_eye="new", right_eye="new"), 2.0))
    assert consumer._take_latest()[1] == 2.0


def test_consumer_preserves_first_frame_until_openxr_is_initialized():
    runtime_q = queue.Queue(maxsize=1)
    runtime_q.put((SimpleNamespace(left_eye="left", right_eye="right"), 1.0))
    shutdown = threading.Event()
    stats = []
    sink = SimpleNamespace(initialized=False)
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append(name),
        sink=sink,
    )
    worker = threading.Thread(target=consumer.run)

    worker.start()
    deadline = time.monotonic() + 1.0
    while "runtime_output_waiting_for_openxr" not in stats:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=1.0)

    assert runtime_q.qsize() == 1
    assert "runtime_output_waiting_for_openxr" in stats


def test_consumer_dispatches_raw_result_to_presenter_without_local_conversion():
    runtime_q = queue.Queue(maxsize=1)
    shutdown = threading.Event()
    calls = []
    stats = []
    runtime_result = SimpleNamespace(left_eye="cuda-left", right_eye="cuda-right")

    class PresenterSink:
        output_ready = True

        def submit_runtime_result(self, result, timestamp):
            calls.append((result, timestamp))

    runtime_q.put((runtime_result, 3.5))
    consumer = VulkanRuntimeOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        source_stat_inc=lambda name, amount=1, **values: stats.append(name),
        sink=PresenterSink(),
    )
    worker = threading.Thread(target=consumer.run)
    worker.start()
    deadline = time.monotonic() + 1.0
    while not calls:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=1.0)

    assert calls == [(runtime_result, 3.5)]
    assert "runtime_output_frames" in stats
