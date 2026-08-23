"""Intel OpenVINO and XPU depth providers."""

from .openvino_native_depth import OpenVINOD3D11DepthProvider
from .openvino_d3d11_native import (
    OpenVINOD3D11Session,
    load_openvino_d3d11_bridge,
    probe_openvino_d3d11_bridge,
)
from .onevpl_d3d11_encoder import (
    OneVPLD3D11SurfaceEncoder,
    probe_onevpl_d3d11,
)
from .d3d11_sbs_surface import D3D11SbsSurface, probe_d3d11_sbs_surface
from .openvino_remote import (
    OpenVINORemoteTensorCapability,
    probe_openvino_remote_tensor,
)
from .pytorch_xpu import (
    DistillAnyDepthBaseXpu,
    GenericAutoDepthXpuProvider,
    GenericTorchXpuDepthProvider,
    TorchXpuDepthProvider,
    create_pytorch_xpu_provider,
    is_xpu_torch_available,
    xpu_device_name,
)

__all__ = [
    "OpenVINOD3D11DepthProvider",
    "OpenVINORemoteTensorCapability",
    "OpenVINOD3D11Session",
    "load_openvino_d3d11_bridge",
    "probe_openvino_d3d11_bridge",
    "probe_openvino_remote_tensor",
    "OneVPLD3D11SurfaceEncoder",
    "probe_onevpl_d3d11",
    "D3D11SbsSurface",
    "probe_d3d11_sbs_surface",
    "TorchXpuDepthProvider",
    "GenericTorchXpuDepthProvider",
    "DistillAnyDepthBaseXpu",
    "GenericAutoDepthXpuProvider",
    "create_pytorch_xpu_provider",
    "is_xpu_torch_available",
    "xpu_device_name",
]
