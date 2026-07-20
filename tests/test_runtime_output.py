from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

from app_runtime.runtime_output import VulkanRuntimeOutputConsumer


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
