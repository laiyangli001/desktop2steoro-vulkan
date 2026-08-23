from __future__ import annotations

from stereo_runtime.providers.intel.openvino_remote import (
    OpenVINORemoteTensorCapability,
    probe_openvino_remote_tensor,
)


def test_openvino_remote_capability_is_explicit_when_runtime_is_missing():
    capability = probe_openvino_remote_tensor()

    assert isinstance(capability, OpenVINORemoteTensorCapability)
    assert capability.directx_remote_tensor is False
    assert capability.zero_copy_ready is False
    assert capability.native_bridge_required is True
    assert capability.reason
