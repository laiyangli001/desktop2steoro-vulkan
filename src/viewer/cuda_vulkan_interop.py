"""NVIDIA CUDA Runtime interop for exportable Vulkan image slots."""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path
from typing import Any

from .vulkan_interop import VulkanInteropCapabilities, VulkanInteropMode
from .vulkan_resources import (
    VulkanExportableImage,
    VulkanExportableSemaphore,
    VulkanImageResource,
)


class CudaVulkanInteropError(RuntimeError):
    pass


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _ExternalHandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", _Win32Handle)]


class _ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _ExternalHandleUnion),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
    ]


class _ChannelFormatDesc(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int),
        ("w", ctypes.c_int),
        ("f", ctypes.c_int),
    ]


class _Extent(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("depth", ctypes.c_size_t),
    ]


class _ExternalMipmappedArrayDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint64),
        ("format_desc", _ChannelFormatDesc),
        ("extent", _Extent),
        ("flags", ctypes.c_uint),
        ("num_levels", ctypes.c_uint),
    ]


class _ExternalSemaphoreHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _ExternalHandleUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _SemaphoreNvSciSync(ctypes.Union):
    _fields_ = [
        ("fence", ctypes.c_void_p),
        ("reserved", ctypes.c_uint64),
    ]


class _SemaphoreSignalParams(ctypes.Structure):
    _fields_ = [
        ("fence_value", ctypes.c_uint64),
        ("nv_sci_sync", _SemaphoreNvSciSync),
        ("keyed_mutex_key", ctypes.c_uint64),
        ("reserved", ctypes.c_uint * 12),
    ]


class _ExternalSemaphoreSignalParams(ctypes.Structure):
    _fields_ = [
        ("params", _SemaphoreSignalParams),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalSemaphoreWaitParams(ctypes.Structure):
    _fields_ = [
        ("params", _SemaphoreSignalParams),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _CudaSlot:
    def __init__(self, target: VulkanExportableImage, external_memory: ctypes.c_void_p, array: ctypes.c_void_p):
        self.target = target
        self.external_memory = external_memory
        self.array = array


class _CudaSemaphore:
    def __init__(self, target: VulkanExportableSemaphore, external: ctypes.c_void_p):
        self.target = target
        self.external = external


class CudaVulkanImageImporter:
    """Import Vulkan-exported memory once and copy CUDA RGBA tensors into it."""

    _CUDA_OPAQUE_FD = 1
    _CUDA_OPAQUE_WIN32 = 2
    _CUDA_ARRAY_COLOR_ATTACHMENT = 0x20
    _CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self, *, cudart_path: str | None = None) -> None:
        self._cudart = self._load_cudart(cudart_path)
        self._slots: dict[int, _CudaSlot] = {}
        self._semaphores: dict[int, _CudaSemaphore] = {}

    @property
    def capabilities(self) -> VulkanInteropCapabilities:
        return VulkanInteropCapabilities(
            producer="nvidia-cuda-runtime",
            mode=VulkanInteropMode.GPU_COPY,
            external_memory=True,
            external_semaphore=all(
                hasattr(self._cudart, name)
                for name in (
                    "cudaImportExternalSemaphore",
                    "cudaSignalExternalSemaphoresAsync",
                    "cudaWaitExternalSemaphoresAsync",
                    "cudaDestroyExternalSemaphore",
                )
            ),
            zero_copy=False,
        )

    @staticmethod
    def _load_cudart(cudart_path: str | None):
        candidates = []
        if cudart_path:
            candidates.append(str(cudart_path))
        env_path = os.environ.get("D2S_CUDART_PATH")
        if env_path:
            candidates.append(env_path)
        package_root = Path(__file__).resolve().parents[1]
        candidates.extend(
            glob.glob(str(package_root / "python3" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin" / "cudart64_*.dll"))
        )
        candidates.extend(glob.glob(str(package_root / "python3" / "Lib" / "site-packages" / "torch" / "lib" / "cudart64_*.dll")))
        candidates.append("cudart64_12.dll")
        for candidate in candidates:
            try:
                lib = ctypes.WinDLL(candidate) if os.name == "nt" else ctypes.CDLL(candidate)
                return CudaVulkanImageImporter._configure_functions(lib)
            except (OSError, AttributeError):
                continue
        raise CudaVulkanInteropError("CUDA Runtime library with external-memory API was not found")

    @staticmethod
    def _configure_functions(lib):
        lib.cudaImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_ExternalMemoryHandleDesc)]
        lib.cudaImportExternalMemory.restype = ctypes.c_int
        lib.cudaExternalMemoryGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(_ExternalMipmappedArrayDesc)]
        lib.cudaExternalMemoryGetMappedMipmappedArray.restype = ctypes.c_int
        lib.cudaGetMipmappedArrayLevel.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        lib.cudaGetMipmappedArrayLevel.restype = ctypes.c_int
        lib.cudaMemcpy2DToArrayAsync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        lib.cudaMemcpy2DToArrayAsync.restype = ctypes.c_int
        lib.cudaDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        lib.cudaDestroyExternalMemory.restype = ctypes.c_int
        import_external = getattr(lib, "cudaImportExternalSemaphore", None)
        signal_external = getattr(lib, "cudaSignalExternalSemaphoresAsync", None)
        wait_external = getattr(lib, "cudaWaitExternalSemaphoresAsync", None)
        destroy_external = getattr(lib, "cudaDestroyExternalSemaphore", None)
        if import_external is not None:
            import_external.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_ExternalSemaphoreHandleDesc),
            ]
            import_external.restype = ctypes.c_int
        if signal_external is not None:
            signal_external.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_ExternalSemaphoreSignalParams),
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            signal_external.restype = ctypes.c_int
        if wait_external is not None:
            wait_external.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(_ExternalSemaphoreWaitParams),
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            wait_external.restype = ctypes.c_int
        if destroy_external is not None:
            destroy_external.argtypes = [ctypes.c_void_p]
            destroy_external.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        return lib

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if int(result) != 0:
            raise CudaVulkanInteropError(f"{operation} failed with CUDA error {int(result)}")

    def register_slot(self, target: VulkanExportableImage) -> VulkanImageResource:
        key = id(target)
        if key in self._slots:
            return target.resource
        if target.resource is None:
            raise CudaVulkanInteropError("exportable target has no Vulkan resource")
        prepare = getattr(target.context, "prepare_external_image_for_cuda", None)
        if not callable(prepare):
            raise CudaVulkanInteropError(
                "Vulkan context cannot establish an external CUDA image layout"
            )
        prepare(target.resource)
        handle = target.export_handle
        desc = _ExternalMemoryHandleDesc(
            type=self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD,
            size=int(target.allocation_size),
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external_memory = ctypes.c_void_p()
        self._check(
            self._cudart.cudaImportExternalMemory(ctypes.byref(external_memory), ctypes.byref(desc)),
            "cudaImportExternalMemory",
        )
        # CUDA receives raw RGBA bytes. Vulkan's sRGB/UNORM interpretation is
        # carried by the image format and does not change this channel layout.
        mapped_desc = _ExternalMipmappedArrayDesc(
            offset=0,
            format_desc=_ChannelFormatDesc(8, 8, 8, 8, 0),
            extent=_Extent(target.width, target.height, 0),
            flags=self._CUDA_ARRAY_COLOR_ATTACHMENT,
            num_levels=1,
        )
        mipmap = ctypes.c_void_p()
        try:
            self._check(
                self._cudart.cudaExternalMemoryGetMappedMipmappedArray(
                    ctypes.byref(mipmap), external_memory, ctypes.byref(mapped_desc)
                ),
                "cudaExternalMemoryGetMappedMipmappedArray",
            )
            array = ctypes.c_void_p()
            self._check(
                self._cudart.cudaGetMipmappedArrayLevel(ctypes.byref(array), mipmap, 0),
                "cudaGetMipmappedArrayLevel",
            )
        except Exception:
            self._cudart.cudaDestroyExternalMemory(external_memory)
            raise
        self._slots[key] = _CudaSlot(target, external_memory, array)
        target.close_export_handle()
        return target.resource

    def copy_tensor(self, tensor: Any, target: VulkanExportableImage, *, stream: int | None = None) -> VulkanImageResource:
        resource = self.register_slot(target)
        if getattr(tensor, "device", None) is None or str(tensor.device.type) != "cuda":
            raise CudaVulkanInteropError("CUDA Vulkan copy requires a CUDA tensor")
        if getattr(tensor, "dtype", None) is None or str(tensor.dtype) != "torch.uint8":
            raise CudaVulkanInteropError("CUDA Vulkan copy requires torch.uint8 RGBA tensor")
        if getattr(tensor, "ndim", 0) != 3 or tuple(tensor.shape) != (target.height, target.width, 4):
            raise CudaVulkanInteropError("CUDA Vulkan copy requires HxWx4 tensor matching target")
        if not bool(tensor.is_contiguous()):
            raise CudaVulkanInteropError("CUDA Vulkan copy requires a contiguous tensor")
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream(device=tensor.device).cuda_stream)
        slot = self._slots[id(target)]
        self._check(
            self._cudart.cudaMemcpy2DToArrayAsync(
                slot.array,
                0,
                0,
                ctypes.c_void_p(int(tensor.data_ptr())),
                target.width * 4,
                target.width * 4,
                target.height,
                self._CUDA_MEMCPY_DEVICE_TO_DEVICE,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMemcpy2DToArrayAsync",
        )
        return resource

    def synchronize(self, *, stream: int | None = None) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        self._check(self._cudart.cudaStreamSynchronize(ctypes.c_void_p(int(stream))), "cudaStreamSynchronize")

    def register_semaphore(self, target: VulkanExportableSemaphore) -> None:
        if not self.capabilities.external_semaphore:
            raise CudaVulkanInteropError("CUDA external semaphore API is unavailable")
        key = id(target)
        if key in self._semaphores:
            return
        handle = target.export_handle
        desc = _ExternalSemaphoreHandleDesc(
            type=(self._CUDA_OPAQUE_WIN32 if os.name == "nt" else self._CUDA_OPAQUE_FD),
            flags=0,
        )
        if os.name == "nt":
            desc.handle.win32.handle = ctypes.c_void_p(int(handle))
        else:
            desc.handle.fd = int(handle)
        external = ctypes.c_void_p()
        self._check(
            self._cudart.cudaImportExternalSemaphore(
                ctypes.byref(external), ctypes.byref(desc)
            ),
            "cudaImportExternalSemaphore",
        )
        target.close_export_handle()
        self._semaphores[key] = _CudaSemaphore(target, external)

    def signal_semaphore(
        self, target: VulkanExportableSemaphore, *, stream: int | None = None
    ) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        semaphore = self._semaphores.get(id(target))
        if semaphore is None:
            raise CudaVulkanInteropError("CUDA external semaphore is not registered")
        params = _ExternalSemaphoreSignalParams(
            params=_SemaphoreSignalParams(
                fence_value=0,
                keyed_mutex_key=0,
            ),
            flags=0,
        )
        semaphore_array = (ctypes.c_void_p * 1)(semaphore.external)
        self._check(
            self._cudart.cudaSignalExternalSemaphoresAsync(
                semaphore_array,
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaSignalExternalSemaphoresAsync",
        )

    def wait_semaphore(
        self, target: VulkanExportableSemaphore, *, stream: int | None = None
    ) -> None:
        if stream is None:
            import torch

            stream = int(torch.cuda.current_stream().cuda_stream)
        wait_external = getattr(self._cudart, "cudaWaitExternalSemaphoresAsync", None)
        if wait_external is None:
            raise CudaVulkanInteropError(
                "CUDA external semaphore wait API is unavailable"
            )
        semaphore = self._semaphores.get(id(target))
        if semaphore is None:
            raise CudaVulkanInteropError("CUDA external semaphore is not registered")
        params = _ExternalSemaphoreWaitParams(
            params=_SemaphoreSignalParams(fence_value=0, keyed_mutex_key=0),
            flags=0,
        )
        semaphore_array = (ctypes.c_void_p * 1)(semaphore.external)
        self._check(
            wait_external(
                semaphore_array,
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream)),
            ),
            "cudaWaitExternalSemaphoresAsync",
        )

    def release_slot(self, target: VulkanExportableImage) -> None:
        slot = self._slots.pop(id(target), None)
        if slot is not None:
            self._check(self._cudart.cudaDestroyExternalMemory(slot.external_memory), "cudaDestroyExternalMemory")

    def close(self) -> None:
        for semaphore in tuple(self._semaphores.values()):
            self._check(
                self._cudart.cudaDestroyExternalSemaphore(semaphore.external),
                "cudaDestroyExternalSemaphore",
            )
        self._semaphores.clear()
        for slot in tuple(self._slots.values()):
            self._check(self._cudart.cudaDestroyExternalMemory(slot.external_memory), "cudaDestroyExternalMemory")
        self._slots.clear()
