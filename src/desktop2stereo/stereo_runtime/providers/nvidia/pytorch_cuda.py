from stereo_runtime.depth_provider import (
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
    GenericTorchDepthProvider,
    TorchDepthProvider,
)

TorchCudaDepthProvider = TorchDepthProvider
GenericTorchCudaDepthProvider = GenericTorchDepthProvider

__all__ = [
    "TorchCudaDepthProvider",
    "GenericTorchCudaDepthProvider",
    "DistillAnyDepthBase518",
    "GenericAutoDepthProvider",
]
