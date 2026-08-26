from __future__ import annotations

import queue
import types

from app_runtime.interop import validate_adapter_luid, validate_resource_share
from capture.capabilities import (
    probe_linux_capture_capabilities,
    probe_windows_capture_capabilities,
)
from capture.types import CapturedFrame, release_native_resource
from stereo_runtime.providers.directml import probe_directml_capabilities
from stereo_runtime.providers.directml_resource import assess_directml_resource
from utils.queue_utils import clear_nonblocking, put_latest


class BorrowedResource:
    adapter_luid = 0x1234
    resource_kind = "d3d11_texture"
    format = "BGRA8"
    width = 16
    height = 8

    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def test_adapter_luid_validation_rejects_missing_and_mismatch():
    assert validate_adapter_luid(0, 0).allowed is False
    mismatch = validate_adapter_luid(0x10, 0x20)
    assert mismatch.allowed is False
    assert "mismatch" in mismatch.reason.lower()
    matched = validate_adapter_luid(0x10, 0x10)
    assert matched.allowed is True
    assert matched.zero_copy is True


def test_resource_share_checks_format_and_dimensions():
    resource = types.SimpleNamespace(
        adapter_luid=0x10,
        format="BGRA8",
        width=16,
        height=8,
    )
    assert validate_resource_share(
        resource,
        0x10,
        expected_format="BGRA8",
        expected_width=16,
        expected_height=8,
    ).allowed
    assert not validate_resource_share(resource, 0x10, expected_format="RGBA8").allowed
    assert not validate_resource_share(resource, 0x20).allowed


def test_queue_helpers_release_borrowed_resources_when_dropped():
    resource = BorrowedResource()
    item = CapturedFrame("frame", 1080, 1.0, native_resource=resource)
    q = queue.Queue(maxsize=1)
    q.put(item)
    put_latest(q, "newest")
    assert resource.released == 1
    clear_nonblocking(q)
    q.put(CapturedFrame("frame2", 1080, 2.0, native_resource=resource))
    clear_nonblocking(q)
    assert resource.released == 2
    assert release_native_resource(item) is True
    assert resource.released == 3


def test_linux_probe_never_claims_native_zero_copy():
    report = probe_linux_capture_capabilities()
    assert report["implemented"]["zero_copy"] is False
    assert report["dmabuf"]["capture_implemented"] is False
    assert report["wayland"]["capture_implemented"] is False


def test_windows_probe_is_explicitly_unavailable_off_windows():
    report = probe_windows_capture_capabilities()
    if report.get("platform") != "Windows":
        assert report["reason"]


def test_directml_probe_is_safe_when_runtime_is_missing():
    report = probe_directml_capabilities()
    assert report["backend"] == "directml"
    assert report["zero_copy"] is False
    assert report["model_operator_support"] in {"not_probed", "representative_only"}


def test_directml_resource_decision_is_explicit_about_cpu_fallback():
    resource = BorrowedResource()
    decision = assess_directml_resource(
        resource,
        consumer_adapter_luid=resource.adapter_luid,
        expected_format="BGRA8",
        expected_width=resource.width,
        expected_height=resource.height,
    )
    assert decision.mode == "cpu_compat"
    assert decision.gpu_to_cpu is True
    assert decision.zero_copy is False
    assert decision.gpu_copy_count == 1


def test_directml_resource_decision_rejects_without_cpu_fallback():
    resource = BorrowedResource()
    decision = assess_directml_resource(
        resource,
        consumer_adapter_luid=resource.adapter_luid,
        allow_cpu_fallback=False,
    )
    assert decision.allowed is False
    assert decision.mode == "rejected"


def test_directml_resource_decision_accepts_shared_handle_only_after_luid_match():
    resource = types.SimpleNamespace(
        adapter_luid=0x10,
        format="BGRA8",
        width=16,
        height=8,
        shared_handle=123,
    )
    decision = assess_directml_resource(
        resource,
        consumer_adapter_luid=0x10,
        expected_format="BGRA8",
        expected_width=16,
        expected_height=8,
    )
    assert decision.mode == "shared"
    assert decision.gpu_to_cpu is False
    assert decision.zero_copy is False
