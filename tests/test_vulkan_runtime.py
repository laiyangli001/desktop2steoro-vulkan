from __future__ import annotations

import pytest
from types import SimpleNamespace

import app_runtime.vulkan_runtime as runtime_module
from app_runtime.vulkan_runtime import (
    VulkanDeviceLostError,
    VulkanRuntimeConfig,
    VulkanRuntimeSession,
)


def test_vulkan_runtime_config_requires_positive_dimensions():
    assert VulkanRuntimeConfig(1280, 720).width == 1280
    with pytest.raises(ValueError, match="dimensions must be positive"):
        VulkanRuntimeConfig(0, 720)


def test_vulkan_runtime_session_delegates_image_submission(monkeypatch):
    calls = []

    class FakePass:
        def __init__(self, context, **kwargs):
            calls.append(("create", context, kwargs))

        def submit(self, source, output, **kwargs):
            calls.append(("submit", source, output, kwargs))
            return 21

        def close(self):
            calls.append(("pass_close",))

    class FakeContext:
        def wait_idle(self):
            calls.append(("wait_idle",))

        def close(self):
            calls.append(("context_close",))

    monkeypatch.setattr(runtime_module, "VulkanImageCopyPass", FakePass)
    context = FakeContext()
    session = VulkanRuntimeSession(context, VulkanRuntimeConfig(16, 9))

    assert session.submit_image_pair(
        "source",
        "output",
        frame_id=5,
        config_version=2,
        ready_timeline=8,
    ) == 21
    session.close()

    assert calls[0][0] == "create"
    assert calls[1] == (
        "submit",
        "source",
        "output",
        {"frame_id": 5, "config_version": 2, "ready_timeline": 8},
    )
    assert calls[-2:] == [("wait_idle",), ("pass_close",)]
    assert all(call[0] != "context_close" for call in calls)


def test_vulkan_runtime_session_submits_external_image_pair(monkeypatch):
    calls = []

    class FakePass:
        def __init__(self, context, **kwargs):
            pass

        def submit(self, source, output, **kwargs):
            calls.append((source, output, kwargs))
            return 9

        def close(self):
            pass

    class FakeContext:
        def wait_idle(self):
            pass

    monkeypatch.setattr(runtime_module, "VulkanImageCopyPass", FakePass)
    context = FakeContext()
    session = VulkanRuntimeSession(context, VulkanRuntimeConfig(8, 4))
    source = SimpleNamespace(context=context, external=True)
    output = SimpleNamespace(context=context, external=True)

    assert session.submit_external_image_pair(
        source,
        output,
        frame_id=3,
        config_version=4,
        ready_timeline=7,
    ) == 9
    assert calls[0][2]["ready_timeline"] == 7


def test_vulkan_runtime_session_resize_replaces_pass_after_idle(monkeypatch):
    calls = []

    class FakePass:
        def __init__(self, context, **kwargs):
            calls.append(("create", kwargs["width"], kwargs["height"]))

        def close(self):
            calls.append(("pass_close",))

    class FakeContext:
        def wait_idle(self):
            calls.append(("wait_idle",))

    monkeypatch.setattr(runtime_module, "VulkanImageCopyPass", FakePass)
    session = VulkanRuntimeSession(FakeContext(), VulkanRuntimeConfig(16, 9))
    session.resize(32, 18)

    assert session.config.width == 32
    assert session.config.height == 18
    assert calls == [
        ("create", 16, 9),
        ("create", 32, 18),
        ("wait_idle",),
        ("pass_close",),
    ]


def test_vulkan_runtime_session_resize_failure_keeps_previous_pass(monkeypatch):
    class FakePass:
        created = 0

        def __init__(self, context, **kwargs):
            type(self).created += 1
            if kwargs["width"] == 32:
                raise RuntimeError("new pipeline failed")

        def close(self):
            pass

    class FakeContext:
        def wait_idle(self):
            pass

    monkeypatch.setattr(runtime_module, "VulkanImageCopyPass", FakePass)
    session = VulkanRuntimeSession(FakeContext(), VulkanRuntimeConfig(16, 9))
    previous_pass = session.image_copy_pass

    with pytest.raises(RuntimeError, match="new pipeline failed"):
        session.resize(32, 18)

    assert session.image_copy_pass is previous_pass
    assert session.config.width == 16
    assert session.config.height == 9


def test_vulkan_runtime_session_marks_device_lost_and_rejects_future_submit(monkeypatch):
    class FakePass:
        def __init__(self, context, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            raise RuntimeError("VK_ERROR_DEVICE_LOST")

        def close(self):
            pass

    class FakeContext:
        def wait_idle(self):
            pass

    monkeypatch.setattr(runtime_module, "VulkanImageCopyPass", FakePass)
    session = VulkanRuntimeSession(FakeContext(), VulkanRuntimeConfig(1, 1))

    with pytest.raises(VulkanDeviceLostError, match="recreate"):
        session.submit_image_pair("source", "output", frame_id=1, config_version=1)
    assert session.device_lost is True
    assert session.last_error == "VK_ERROR_DEVICE_LOST"
    with pytest.raises(VulkanDeviceLostError, match="recreate"):
        session.submit_image_pair("source", "output", frame_id=2, config_version=1)


def test_vulkan_runtime_session_preserves_non_device_errors(monkeypatch):
    class FakePass:
        def __init__(self, context, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            raise RuntimeError("descriptor budget exhausted")

        def close(self):
            pass

    class FakeContext:
        def wait_idle(self):
            pass

    monkeypatch.setattr(runtime_module, "VulkanImageCopyPass", FakePass)
    session = VulkanRuntimeSession(FakeContext(), VulkanRuntimeConfig(1, 1))
    with pytest.raises(RuntimeError, match="descriptor budget"):
        session.submit_image_pair("source", "output", frame_id=1, config_version=1)
    assert session.device_lost is False
