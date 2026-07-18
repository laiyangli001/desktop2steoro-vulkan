"""AMD ROCm and DirectML depth providers."""

from .migraphx import (
    MIGraphXDepthProvider,
    MIGraphXEngine,
    build_migraphx_graph,
    create_migraphx_rocm_provider,
    is_migraphx_available,
)
from .pytorch_rocm import (
    DistillAnyDepthBaseRocm,
    GenericAutoDepthRocmProvider,
    GenericTorchRocmDepthProvider,
    TorchRocmDepthProvider,
    create_pytorch_rocm_provider,
    is_rocm_torch_available,
    rocm_device_name,
)

__all__ = [
    "MIGraphXDepthProvider",
    "MIGraphXEngine",
    "build_migraphx_graph",
    "create_migraphx_rocm_provider",
    "is_migraphx_available",
    "TorchRocmDepthProvider",
    "GenericTorchRocmDepthProvider",
    "DistillAnyDepthBaseRocm",
    "GenericAutoDepthRocmProvider",
    "create_pytorch_rocm_provider",
    "is_rocm_torch_available",
    "rocm_device_name",
]
