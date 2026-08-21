"""Optional native AV_PIX_FMT_VULKAN bridge loader.

The host-upload Vulkan path must remain available when the native bridge is
not installed. This module only accepts the versioned narrow ABI; it never
silently treats a random DLL as a zero-copy encoder.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


EXPECTED_ABI_VERSION = 5
REQUIRED_SYMBOLS = (
    "d2s_vulkan_ffmpeg_bridge_abi_version",
    "d2s_vulkan_ffmpeg_bridge_probe",
    "d2s_vulkan_ffmpeg_encoder_create",
    "d2s_vulkan_ffmpeg_encoder_acquire_frame",
    "d2s_vulkan_ffmpeg_encoder_acquire_rgba_frame",
    "d2s_vulkan_ffmpeg_encoder_release_rgba_frame",
    "d2s_vulkan_ffmpeg_encoder_submit_frame",
    "d2s_vulkan_ffmpeg_encoder_submit_image",
    "d2s_vulkan_ffmpeg_encoder_read_packet",
    "d2s_vulkan_ffmpeg_encoder_flush",
    "d2s_vulkan_ffmpeg_encoder_destroy",
)


class VulkanVideoFrame(ctypes.Structure):
    """Opaque-handle description of one native FFmpeg NV12 frame slot."""

    _fields_ = [
        ("image", ctypes.c_void_p * 2),
        ("memory", ctypes.c_void_p * 2),
        ("memory_size", ctypes.c_uint64 * 2),
        ("memory_offset", ctypes.c_int64 * 2),
        ("format", ctypes.c_uint32 * 2),
        ("layout", ctypes.c_uint32 * 2),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("plane_count", ctypes.c_uint32),
        ("external_memory_handle", ctypes.c_int64 * 2),
        ("external_semaphore_handle", ctypes.c_int64 * 2),
        ("semaphore_value", ctypes.c_uint64 * 2),
        ("slot_id", ctypes.c_uint64),
        ("external_handle_type", ctypes.c_uint32),
    ]

    @property
    def is_split_nv12(self) -> bool:
        """Whether FFmpeg exposed NV12 as two single-plane Vulkan images.

        NVIDIA Vulkan Video requires one multi-plane NV12 image for the encode
        source.  The split representation is useful for diagnostics/CUDA
        import, but must never be submitted to h264_vulkan/hevc_vulkan.
        """
        return int(self.plane_count) == 2 and int(self.format[0]) == 9 and int(self.format[1]) == 16

    @property
    def video_encode_compatible(self) -> bool:
        return int(self.plane_count) == 1 or not self.is_split_nv12


class VulkanNativeEncoder:
    """Typed owner for the native encoder handle and frame-pool ABI."""

    def __init__(self, bridge: "VulkanNativeBridge", handle: int) -> None:
        if not handle:
            raise RuntimeError("native Vulkan FFmpeg encoder creation failed")
        self.bridge = bridge
        self.handle = ctypes.c_void_p(int(handle))
        self._acquired: VulkanVideoFrame | None = None

    def acquire_frame(self) -> VulkanVideoFrame:
        if self._acquired is not None:
            raise RuntimeError("a Vulkan encoder frame is already acquired")
        frame = VulkanVideoFrame()
        result = int(self.bridge._acquire_frame(self.handle, ctypes.byref(frame)))
        if result != 0:
            raise RuntimeError(f"native Vulkan frame acquire failed: {result}")
        if not frame.external_handle_type or not all(frame.external_memory_handle[: frame.plane_count]):
            raise RuntimeError("native Vulkan frame has no CUDA-importable external memory handles")
        if not all(frame.external_semaphore_handle[: frame.plane_count]):
            raise RuntimeError("native Vulkan frame has no CUDA-importable semaphore handles")
        if frame.is_split_nv12:
            self._close_exported_handles(frame)
            self._acquired = None
            raise RuntimeError(
                "Vulkan Video frame is split into R8/R8G8 planes; "
                "requires a single multi-plane NV12 image, fallback required"
            )
        self._acquired = frame
        return frame

    def acquire_rgba_frame(self) -> VulkanVideoFrame:
        if self._acquired is not None:
            raise RuntimeError("a Vulkan encoder frame is already acquired")
        frame = VulkanVideoFrame()
        result = int(self.bridge._acquire_rgba_frame(self.handle, ctypes.byref(frame)))
        if result != 0:
            raise RuntimeError(f"native Vulkan RGBA frame acquire failed: {result}")
        if frame.plane_count != 1 or not frame.external_handle_type or not frame.external_memory_handle[0] or not frame.external_semaphore_handle[0]:
            self._close_exported_handles(frame)
            raise RuntimeError("native Vulkan RGBA frame is not a single external image")
        self._acquired = frame
        return frame

    def release_rgba_frame(self, ready_value: int) -> None:
        if int(ready_value) <= 0:
            raise ValueError("RGBA frame release requires a positive ready timeline value")
        result = int(
            self.bridge._release_rgba_frame(self.handle, int(ready_value))
        )
        if result != 0:
            raise RuntimeError(f"native Vulkan RGBA frame release failed: {result}")
        self._acquired = None

    @staticmethod
    def _close_exported_handles(frame: VulkanVideoFrame) -> None:
        for index in range(int(frame.plane_count)):
            for handle in (int(frame.external_memory_handle[index]), int(frame.external_semaphore_handle[index])):
                if not handle:
                    continue
                if os.name == "nt":
                    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
                else:
                    os.close(handle)

    def submit_frame(
        self,
        frame: VulkanVideoFrame,
        *,
        timestamp: int,
        ready_semaphore: int | None = None,
        ready_value: int = 0,
    ) -> None:
        if self._acquired is None:
            raise RuntimeError("no Vulkan encoder frame is acquired")
        if not ready_semaphore:
            raise RuntimeError(
                "Vulkan frame submission requires a producer-ready semaphore; "
                "unsynchronized GPU writes are rejected"
            )
        result = int(
            self.bridge._submit_frame(
                self.handle,
                ctypes.byref(frame),
                ctypes.c_void_p(int(ready_semaphore)),
                int(ready_value),
                int(timestamp),
            )
        )
        if result != 0:
            raise RuntimeError(f"native Vulkan frame submit failed: {result}")
        self._acquired = None

    def read_packet(self, capacity: int = 4 * 1024 * 1024) -> bytes | None:
        output = ctypes.create_string_buffer(int(capacity))
        timestamp = ctypes.c_int64()
        keyframe = ctypes.c_int()
        result = int(
            self.bridge._read_packet(
                self.handle,
                output,
                int(capacity),
                ctypes.byref(timestamp),
                ctypes.byref(keyframe),
            )
        )
        if result == 0:
            return None
        if result < 0:
            raise RuntimeError(f"native Vulkan packet read failed: {result}")
        return output.raw[:result]

    def flush(self) -> None:
        result = int(self.bridge._flush(self.handle))
        if result != 0:
            raise RuntimeError(f"native Vulkan encoder flush failed: {result}")

    def close(self) -> None:
        if self.handle:
            self.bridge._destroy(self.handle)
            self.handle = ctypes.c_void_p()
            self._acquired = None


class VulkanNativeBridge:
    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        missing = [name for name in REQUIRED_SYMBOLS if not hasattr(library, name)]
        if missing:
            raise RuntimeError("missing ABI symbols: " + ", ".join(missing))
        version = int(library.d2s_vulkan_ffmpeg_bridge_abi_version())
        if version != EXPECTED_ABI_VERSION:
            raise RuntimeError(f"unsupported ABI version: {version}")
        probe = library.d2s_vulkan_ffmpeg_bridge_probe
        probe.argtypes = [ctypes.c_char_p, ctypes.c_int]
        probe.restype = ctypes.c_int
        message = ctypes.create_string_buffer(512)
        if int(probe(message, len(message))) != 1:
            detail = message.value.decode("utf-8", errors="replace")
            raise RuntimeError(detail or "native Vulkan FFmpeg probe failed")
        self._create = library.d2s_vulkan_ffmpeg_encoder_create
        self._create.restype = ctypes.c_void_p
        self._create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._acquire_frame = library.d2s_vulkan_ffmpeg_encoder_acquire_frame
        self._acquire_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(VulkanVideoFrame)]
        self._acquire_frame.restype = ctypes.c_int
        self._acquire_rgba_frame = library.d2s_vulkan_ffmpeg_encoder_acquire_rgba_frame
        self._acquire_rgba_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(VulkanVideoFrame)]
        self._acquire_rgba_frame.restype = ctypes.c_int
        self._release_rgba_frame = library.d2s_vulkan_ffmpeg_encoder_release_rgba_frame
        self._release_rgba_frame.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._release_rgba_frame.restype = ctypes.c_int
        self._submit_frame = library.d2s_vulkan_ffmpeg_encoder_submit_frame
        self._submit_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VulkanVideoFrame),
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int64,
        ]
        self._submit_frame.restype = ctypes.c_int
        self._read_packet = library.d2s_vulkan_ffmpeg_encoder_read_packet
        self._read_packet.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._read_packet.restype = ctypes.c_int
        self._flush = library.d2s_vulkan_ffmpeg_encoder_flush
        self._flush.argtypes = [ctypes.c_void_p]
        self._flush.restype = ctypes.c_int
        self._destroy = library.d2s_vulkan_ffmpeg_encoder_destroy
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None

    def create_encoder(
        self,
        *,
        instance: int = 0,
        physical_device: int = 0,
        device: int = 0,
        queue: int = 0,
        queue_family: int = -1,
        width: int,
        height: int,
        fps: int,
        target_bitrate: int,
        peak_bitrate: int,
        hevc: bool = False,
    ) -> VulkanNativeEncoder:
        supplied = (int(instance), int(physical_device), int(device), int(queue))
        if any(supplied) and not all(supplied):
            raise ValueError(
                "Vulkan bridge external-device mode requires instance, physical_device, "
                "device and queue together"
            )
        handle = self._create(
            ctypes.c_void_p(int(instance)),
            ctypes.c_void_p(int(physical_device)),
            ctypes.c_void_p(int(device)),
            ctypes.c_void_p(int(queue)),
            int(queue_family),
            int(width),
            int(height),
            int(fps),
            int(target_bitrate),
            int(peak_bitrate),
            int(bool(hevc)),
        )
        return VulkanNativeEncoder(self, int(handle or 0))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "VulkanNativeBridge":
        configured = path or os.environ.get("D2S_VULKAN_FFMPEG_BRIDGE")
        if not configured:
            raise FileNotFoundError("D2S_VULKAN_FFMPEG_BRIDGE is not configured")
        candidate = Path(configured)
        if not candidate.is_file():
            raise FileNotFoundError(f"Vulkan FFmpeg bridge not found: {candidate}")
        return cls(ctypes.CDLL(str(candidate)))
