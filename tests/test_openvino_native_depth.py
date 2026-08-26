from types import SimpleNamespace

import pytest

from stereo_runtime.providers.intel.openvino_native_depth import (
    OpenVINOD3D11DepthProvider,
)


def _provider(session_luid=0x10):
    provider = object.__new__(OpenVINOD3D11DepthProvider)
    provider.session = SimpleNamespace(adapter_luid=session_luid)
    return provider


def test_openvino_native_rejects_missing_adapter_luid():
    resource = SimpleNamespace(
        texture=object(),
        adapter_luid=0,
        width=16,
        height=8,
        format="BGRA8",
    )
    with pytest.raises(RuntimeError, match="LUID is required"):
        _provider().predict_native(resource)


def test_openvino_native_rejects_adapter_mismatch():
    resource = SimpleNamespace(
        texture=object(),
        adapter_luid=0x20,
        width=16,
        height=8,
        format="BGRA8",
    )
    with pytest.raises(RuntimeError, match="LUID mismatch"):
        _provider().predict_native(resource)


def test_openvino_native_rejects_unsupported_format():
    resource = SimpleNamespace(
        texture=object(),
        adapter_luid=0x10,
        width=16,
        height=8,
        format="RGBA8",
    )
    with pytest.raises(RuntimeError, match="requires BGRA8"):
        _provider().predict_native(resource)
