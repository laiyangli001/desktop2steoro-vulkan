"""NVIDIA CUDA / ONNX / TensorRT depth providers."""

from .onnx_cuda import DistillAnyDepthBaseOnnxCuda, OnnxCudaDepthProvider
from .pytorch_cuda import (
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
    GenericTorchCudaDepthProvider,
    TorchCudaDepthProvider,
)
from .tensorrt_native import (
    DistillAnyDepthBaseNativeTensorRt,
    NativeTensorRtDepthProvider,
    build_native_tensorrt_engine,
)
from .tensorrt_ort import DistillAnyDepthBaseTensorRtOrt, TensorRtOrtDepthProvider

__all__ = [
    "DistillAnyDepthBase518",
    "GenericAutoDepthProvider",
    "TorchCudaDepthProvider",
    "GenericTorchCudaDepthProvider",
    "OnnxCudaDepthProvider",
    "NativeTensorRtDepthProvider",
    "TensorRtOrtDepthProvider",
    "DistillAnyDepthBaseOnnxCuda",
    "DistillAnyDepthBaseNativeTensorRt",
    "DistillAnyDepthBaseTensorRtOrt",
    "build_native_tensorrt_engine",
]
