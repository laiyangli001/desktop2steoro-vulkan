"""Intel OpenVINO/D3D11 zero-copy capability contract.

The Python OpenVINO package can report GPU availability, but wrapping a
Windows ID3D11Texture2D as a GPU RemoteTensor is a native C++ integration
point. This module deliberately reports that distinction instead of claiming
Python-only zero-copy support.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from .openvino_d3d11_native import probe_openvino_d3d11_bridge


@dataclass(frozen=True)
class OpenVINORemoteTensorCapability:
    runtime_available: bool
    directx_remote_tensor: bool
    native_bridge_required: bool
    reason: str | None = None
    native_bridge_available: bool = False

    @property
    def zero_copy_ready(self) -> bool:
        return self.runtime_available and self.directx_remote_tensor


def probe_openvino_remote_tensor() -> OpenVINORemoteTensorCapability:
    if os.name != "nt":
        return OpenVINORemoteTensorCapability(
            runtime_available=False,
            directx_remote_tensor=False,
            native_bridge_required=True,
            reason="Intel D3D11 RemoteTensor path requires Windows",
        )
    bridge = probe_openvino_d3d11_bridge()
    if bridge["available"]:
        bridge_ready = bool(bridge.get("nv12_surface")) and bool(
            bridge.get("bgra_to_nv12")
        )
        return OpenVINORemoteTensorCapability(
            runtime_available=True,
            directx_remote_tensor=bridge_ready,
            native_bridge_required=not bridge_ready,
            native_bridge_available=True,
            reason=None if bridge_ready else str(bridge.get("reason")),
        )
    if importlib.util.find_spec("openvino") is None:
        return OpenVINORemoteTensorCapability(
            runtime_available=False,
            directx_remote_tensor=False,
            native_bridge_required=True,
            reason="openvino Python package is not installed",
        )
    return OpenVINORemoteTensorCapability(
        runtime_available=True,
        directx_remote_tensor=False,
        native_bridge_required=True,
        reason="D3D11 RemoteTensor wrapping must be provided by the native bridge",
    )


__all__ = [
    "OpenVINORemoteTensorCapability",
    "probe_openvino_remote_tensor",
]
