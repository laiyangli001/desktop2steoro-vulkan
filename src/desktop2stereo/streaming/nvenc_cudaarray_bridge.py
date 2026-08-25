"""ctypes adapter for the native NVENC CUDAARRAY encoder bridge."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
from typing import Any


_ABI_VERSION = 1
_DLL_NAME = "d2s_nvenc_cudaarray_bridge.dll"


def _candidate_libraries() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("D2S_NVENC_CUDAARRAY_BRIDGE", "").strip()
    if override:
        candidates.append(Path(override))
    feature_dir = Path(__file__).with_name("nvenc_cudaarray_bridge")
    candidates.append(feature_dir / _DLL_NAME)
    candidates.append(Path(__file__).resolve().parents[3] / "native" / "nvenc_cudaarray_bridge" / "build" / "Release" / _DLL_NAME)
    return candidates


def _configure_api(library: Any) -> None:
    library.d2s_nvenc_cudaarray_abi_version.restype = ctypes.c_uint32
    library.d2s_nvenc_cudaarray_probe.restype = ctypes.c_int32
    library.d2s_nvenc_cudaarray_create.restype = ctypes.c_void_p
    library.d2s_nvenc_cudaarray_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_uint64,
    ]
    library.d2s_nvenc_cudaarray_submit.restype = ctypes.c_int32
    library.d2s_nvenc_cudaarray_submit.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    library.d2s_nvenc_cudaarray_read_packet.restype = ctypes.c_int32
    library.d2s_nvenc_cudaarray_read_packet.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.d2s_nvenc_cudaarray_flush.restype = ctypes.c_int32
    library.d2s_nvenc_cudaarray_flush.argtypes = [ctypes.c_void_p]
    library.d2s_nvenc_cudaarray_destroy.restype = None
    library.d2s_nvenc_cudaarray_destroy.argtypes = [ctypes.c_void_p]
    library.d2s_nvenc_cudaarray_last_error.restype = ctypes.c_char_p


def _last_error(library: Any) -> str:
    raw = library.d2s_nvenc_cudaarray_last_error()
    if not raw:
        return "native NVENC CUDAARRAY operation failed"
    return raw.decode("utf-8", errors="replace")


def load_nvenc_cudaarray_bridge() -> Any:
    if platform.system() != "Windows":
        raise RuntimeError("native NVENC CUDAARRAY bridge is Windows-only")
    errors: list[str] = []
    for candidate in _candidate_libraries():
        try:
            library = ctypes.WinDLL(str(candidate))
            _configure_api(library)
            abi = int(library.d2s_nvenc_cudaarray_abi_version())
            if abi != _ABI_VERSION:
                raise RuntimeError(
                    f"ABI mismatch: expected {_ABI_VERSION}, bridge reports {abi}"
                )
            if int(library.d2s_nvenc_cudaarray_probe()) != 0:
                raise RuntimeError(_last_error(library))
            return library
        except (OSError, RuntimeError) as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no candidate library"
    raise RuntimeError(f"{_DLL_NAME} unavailable: {detail}")


class NvencCudaArrayEncoder:
    """Encode one persistent CUDA array registered directly with NVENC."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fps: int,
        bitrate: int,
        hevc: bool,
        cuda_array: int,
        library: Any | None = None,
    ) -> None:
        self._library = library or load_nvenc_cudaarray_bridge()
        self._handle = self._library.d2s_nvenc_cudaarray_create(
            int(width),
            int(height),
            max(1, int(fps)),
            max(1, int(bitrate)),
            int(bool(hevc)),
            int(cuda_array),
        )
        if not self._handle:
            raise RuntimeError(_last_error(self._library))
        self._cuda_array = int(cuda_array)
        self._timestamp = 0
        self._closed = False

    def encode(self, cuda_array: int) -> bytes:
        if self._closed or not self._handle:
            raise RuntimeError("native NVENC CUDAARRAY encoder is closed")
        # A bridge instance owns one registered array for its entire lifetime.
        # The argument is retained so it can share the packet-output adapter
        # used by PyNvVideoCodec while validating accidental surface changes.
        if int(cuda_array) != self._cuda_array:
            raise ValueError("native NVENC encoder cannot switch CUDA arrays")
        status = int(
            self._library.d2s_nvenc_cudaarray_submit(
                self._handle, ctypes.c_int64(self._timestamp)
            )
        )
        if status != 0:
            raise RuntimeError(_last_error(self._library))
        self._timestamp += 1
        return self._drain_packets()

    def _drain_packets(self) -> bytes:
        output = bytearray()
        while True:
            size = ctypes.c_size_t()
            status = int(
                self._library.d2s_nvenc_cudaarray_read_packet(
                    self._handle, None, 0, ctypes.byref(size)
                )
            )
            if status < 0:
                raise RuntimeError(_last_error(self._library))
            if status == 0 or size.value == 0:
                break
            buffer = (ctypes.c_uint8 * size.value)()
            copied = ctypes.c_size_t()
            status = int(
                self._library.d2s_nvenc_cudaarray_read_packet(
                    self._handle,
                    buffer,
                    size.value,
                    ctypes.byref(copied),
                )
            )
            if status < 0:
                raise RuntimeError(_last_error(self._library))
            output.extend(bytes(buffer[: copied.value]))
        return bytes(output)

    def flush(self) -> bytes:
        if self._closed or not self._handle:
            return b""
        if int(self._library.d2s_nvenc_cudaarray_flush(self._handle)) != 0:
            raise RuntimeError(_last_error(self._library))
        return self._drain_packets()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle:
            self._library.d2s_nvenc_cudaarray_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "NvencCudaArrayEncoder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
