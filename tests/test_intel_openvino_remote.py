from __future__ import annotations

import importlib.util

import pytest

import stereo_runtime.providers.intel.openvino_remote as openvino_remote
from stereo_runtime.providers.intel.openvino_remote import (
    OpenVINORemoteTensorCapability,
    probe_openvino_remote_tensor,
)


def test_openvino_remote_capability_is_explicit_when_runtime_is_missing(monkeypatch):
    """Keep the missing-runtime contract independent of installed CI DLLs."""
    monkeypatch.setattr(
        openvino_remote,
        "probe_openvino_d3d11_bridge",
        lambda: {"available": False, "reason": "test bridge unavailable"},
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    capability = probe_openvino_remote_tensor()

    assert isinstance(capability, OpenVINORemoteTensorCapability)
    assert capability.directx_remote_tensor is False
    assert capability.zero_copy_ready is False
    assert capability.native_bridge_required is True
    assert capability.reason
