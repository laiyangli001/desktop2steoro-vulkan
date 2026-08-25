"""OpenGL image fallback for the advanced streaming path.

The backend has two deliberately separate implementations:

* NVIDIA + CUDA/OpenGL interop or AMD + HIP/OpenGL interop: the GPU tensor
  is uploaded to and read back from an OpenGL RGBA8 texture through the
  vendor graphics API. The round trip stays on the GPU and returns a GPU
  tensor for the PyNvVideoCodec/NVENC or AMF encoder. No raw RGB frame crosses
  the CPU.
* Other/unsupported systems: a hidden OpenGL context, RGBA8 texture and PBO
  ring are still validated, then the existing host-upload FFmpeg path is used.

The vendor interop path contains GPU copies between linear GPU memory and the
OpenGL array; it is therefore GPU-only but not strict zero-copy.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpenGLFallbackCapabilities:
    available: bool
    context_api: str
    texture_format: str
    pbo_count: int
    fence_supported: bool
    cuda_gl_interop: bool
    gpu_to_cpu: bool
    zero_copy: bool
    gpu_copy_count: int = 0
    detail: str = ""
    framebuffer_supported: bool = False
    hip_gl_interop: bool = False
    interop_mode: str = "none"


class _CudaOpenGLInterop:
    """Small CUDA Runtime graphics-interoperability adapter.

    CUDA graphics resources are registered once for the OpenGL texture. Each
    frame maps the resource, copies CUDA RGBA bytes into the mapped array,
    unmaps it, then maps it again and copies the array into a CUDA tensor.
    Both copies are device-to-device; the returned tensor remains on CUDA.
    """

    _CUDA_SUCCESS = 0
    _CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
    # CUDA arrays passed to NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY must expose
    # surface load/store capability. This is the runtime equivalent of
    # cudaGraphicsRegisterFlagsSurfaceLoadStore.
    _CUDA_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4

    def __init__(self, *, texture: int, width: int, height: int, gl_target: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self._texture = int(texture)
        self._resource = ctypes.c_void_p()
        self._mapped_resources = None
        self._mapped_array = ctypes.c_void_p()
        self._mapped_stream = 0
        self._cudart = self._load_cudart()
        self._configure_api()
        self._check(
            self._cudart.cudaGraphicsGLRegisterImage(
                ctypes.byref(self._resource),
                ctypes.c_uint(self._texture),
                ctypes.c_uint(int(gl_target)),
                ctypes.c_uint(self._CUDA_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST),
            ),
            "cudaGraphicsGLRegisterImage",
        )

    @staticmethod
    def _load_cudart() -> Any:
        candidates: list[str] = []
        if platform.system() == "Windows":
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                candidates.append(str(Path(cuda_path) / "bin" / "cudart64_*.dll"))
            candidates.extend(
                [
                    str(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin"),
                    str(Path(__file__).resolve().parents[2] / "python3" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin"),
                ]
            )
            for pattern in candidates[:]:
                if "*" in pattern:
                    parent = Path(pattern).parent
                    candidates.extend(str(path) for path in parent.glob(Path(pattern).name))
            candidates.extend(["cudart64_12.dll", "cudart64_11.dll"])
        else:
            candidates.extend(
                [
                    "libcudart.so",
                    "libcudart.so.12",
                    "libcudart.so.11.0",
                ]
            )
        for candidate in candidates:
            try:
                if platform.system() == "Windows":
                    return ctypes.WinDLL(candidate)
                return ctypes.CDLL(candidate)
            except (OSError, TypeError):
                continue
        located = ctypes.util.find_library("cudart")
        if located:
            try:
                if platform.system() == "Windows":
                    return ctypes.WinDLL(located)
                return ctypes.CDLL(located)
            except OSError:
                pass
        raise RuntimeError("CUDA Runtime library (cudart) is unavailable")

    def _configure_api(self) -> None:
        lib = self._cudart
        lib.cudaGraphicsGLRegisterImage.restype = ctypes.c_int
        lib.cudaGraphicsGLRegisterImage.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.cudaGraphicsMapResources.restype = ctypes.c_int
        lib.cudaGraphicsMapResources.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
        lib.cudaGraphicsUnmapResources.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cudaGraphicsSubResourceGetMappedArray.restype = ctypes.c_int
        lib.cudaGraphicsSubResourceGetMappedArray.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.cudaMemcpy2DToArrayAsync.restype = ctypes.c_int
        lib.cudaMemcpy2DToArrayAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaMemcpy2DFromArrayAsync.restype = ctypes.c_int
        lib.cudaMemcpy2DFromArrayAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
        lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]

    def _check(self, status: int, operation: str) -> None:
        if int(status) != self._CUDA_SUCCESS:
            get_error = getattr(self._cudart, "cudaGetErrorString", None)
            detail = ""
            if get_error is not None:
                get_error.restype = ctypes.c_char_p
                get_error.argtypes = [ctypes.c_int]
                message = get_error(int(status))
                detail = (
                    f": {message.decode('utf-8', errors='replace')}"
                    if message
                    else ""
                )
            raise RuntimeError(f"{operation} failed with CUDA error {int(status)}{detail}")

    def _resource_array(self, stream: int) -> tuple[Any, Any]:
        resources = (ctypes.c_void_p * 1)(self._resource.value)
        self._check(
            self._cudart.cudaGraphicsMapResources(
                ctypes.c_uint(1), resources, ctypes.c_void_p(int(stream))
            ),
            "cudaGraphicsMapResources",
        )
        array = ctypes.c_void_p()
        try:
            self._check(
                self._cudart.cudaGraphicsSubResourceGetMappedArray(
                    ctypes.byref(array),
                    self._resource,
                    ctypes.c_uint(0),
                    ctypes.c_uint(0),
                ),
                "cudaGraphicsSubResourceGetMappedArray",
            )
        except Exception:
            self._cudart.cudaGraphicsUnmapResources(
                ctypes.c_uint(1), resources, ctypes.c_void_p(int(stream))
            )
            raise
        return resources, array

    def _unmap(self, resources: Any, stream: int) -> None:
        self._check(
            self._cudart.cudaGraphicsUnmapResources(
                ctypes.c_uint(1), resources, ctypes.c_void_p(int(stream))
            ),
            "cudaGraphicsUnmapResources",
        )

    def copy_to_array(self, rgba: Any) -> int:
        """Copy RGBA8 into a persistently mapped CUDA array for native NVENC."""
        import torch

        if not bool(getattr(rgba, "is_cuda", False)):
            raise ValueError("CUDA/OpenGL interop requires a CUDA tensor")
        if tuple(rgba.shape) != (self.height, self.width, 4) or rgba.dtype != torch.uint8:
            raise ValueError(
                f"CUDA/OpenGL interop requires CUDA RGBA8 {self.width}x{self.height}, "
                f"got shape={tuple(rgba.shape)!r} dtype={rgba.dtype}"
            )
        stream_object = torch.cuda.current_stream(device=rgba.device)
        stream = int(stream_object.cuda_stream)
        if self._mapped_resources is None:
            self._mapped_resources, self._mapped_array = self._resource_array(stream)
            self._mapped_stream = stream
        width_bytes = self.width * 4
        self._check(
            self._cudart.cudaMemcpy2DToArrayAsync(
                self._mapped_array,
                ctypes.c_size_t(0),
                ctypes.c_size_t(0),
                ctypes.c_void_p(int(rgba.data_ptr())),
                ctypes.c_size_t(width_bytes),
                ctypes.c_size_t(width_bytes),
                ctypes.c_size_t(self.height),
                ctypes.c_int(self._CUDA_MEMCPY_DEVICE_TO_DEVICE),
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpy2DToArrayAsync",
        )
        # NVENC submits through its own queue. It cannot infer the dependency
        # from PyTorch's current stream, so make the hand-off explicit.
        stream_object.synchronize()
        return int(self._mapped_array.value or 0)

    def release_persistent_array(self) -> None:
        if self._mapped_resources is None:
            return
        resources = self._mapped_resources
        stream = self._mapped_stream
        self._mapped_resources = None
        self._mapped_array = ctypes.c_void_p()
        self._mapped_stream = 0
        self._unmap(resources, stream)

    def roundtrip(self, rgba: Any) -> Any:
        import torch

        if not bool(getattr(rgba, "is_cuda", False)):
            raise ValueError("CUDA/OpenGL interop requires a CUDA tensor")
        if tuple(rgba.shape) != (self.height, self.width, 4) or rgba.dtype != torch.uint8:
            raise ValueError(
                f"CUDA/OpenGL interop requires CUDA RGBA8 {self.width}x{self.height}, "
                f"got shape={tuple(rgba.shape)!r} dtype={rgba.dtype}"
            )
        # PyNvVideoCodec needs the historical mapped-array -> linear
        # tensor path. Release a persistent native-NVENC mapping before using it.
        self.release_persistent_array()
        stream = int(torch.cuda.current_stream(device=rgba.device).cuda_stream)
        width_bytes = self.width * 4
        resources, array = self._resource_array(stream)
        try:
            self._check(
                self._cudart.cudaMemcpy2DToArrayAsync(
                    array,
                    ctypes.c_size_t(0),
                    ctypes.c_size_t(0),
                    ctypes.c_void_p(int(rgba.data_ptr())),
                    ctypes.c_size_t(width_bytes),
                    ctypes.c_size_t(width_bytes),
                    ctypes.c_size_t(self.height),
                    ctypes.c_int(self._CUDA_MEMCPY_DEVICE_TO_DEVICE),
                    ctypes.c_void_p(stream),
                ),
                "cudaMemcpy2DToArrayAsync",
            )
        finally:
            self._unmap(resources, stream)

        output = torch.empty_like(rgba)
        resources, array = self._resource_array(stream)
        try:
            self._check(
                self._cudart.cudaMemcpy2DFromArrayAsync(
                    ctypes.c_void_p(int(output.data_ptr())),
                    ctypes.c_size_t(width_bytes),
                    array,
                    ctypes.c_size_t(0),
                    ctypes.c_size_t(0),
                    ctypes.c_size_t(width_bytes),
                    ctypes.c_size_t(self.height),
                    ctypes.c_int(self._CUDA_MEMCPY_DEVICE_TO_DEVICE),
                    ctypes.c_void_p(stream),
                ),
                "cudaMemcpy2DFromArrayAsync",
            )
        finally:
            self._unmap(resources, stream)
        return output

    def close(self) -> None:
        if self._mapped_resources is not None:
            try:
                self.release_persistent_array()
            except Exception:
                pass
        if self._resource and self._resource.value:
            status = self._cudart.cudaGraphicsUnregisterResource(self._resource)
            if int(status) != self._CUDA_SUCCESS:
                # Context teardown can invalidate the resource first. Do not
                # hide the primary streaming failure during cleanup.
                pass
            self._resource = ctypes.c_void_p()


class _HipRuntimeAlias:
    """Expose HIP Runtime graphics symbols under the CUDA-shaped adapter API."""

    def __init__(self, library: Any) -> None:
        self._library = library

    def __getattr__(self, name: str) -> Any:
        if name.startswith("cuda"):
            name = "hip" + name[4:]
        return getattr(self._library, name)


class _HipOpenGLInterop(_CudaOpenGLInterop):
    """HIP equivalent of the CUDA/OpenGL graphics-resource adapter."""

    @staticmethod
    def _load_cudart() -> Any:
        candidates: list[str] = []
        hip_path = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH")
        if platform.system() == "Windows":
            if hip_path:
                candidates.append(str(Path(hip_path) / "bin" / "amdhip64.dll"))
            candidates.append("amdhip64.dll")
        else:
            if hip_path:
                candidates.append(str(Path(hip_path) / "lib" / "libamdhip64.so"))
            candidates.extend(["libamdhip64.so", "libamdhip64.so.6"])
        for candidate in candidates:
            try:
                library = (
                    ctypes.WinDLL(candidate)
                    if platform.system() == "Windows"
                    else ctypes.CDLL(candidate)
                )
                return _HipRuntimeAlias(library)
            except (OSError, TypeError):
                continue
        located = ctypes.util.find_library("amdhip64")
        if located:
            library = (
                ctypes.WinDLL(located)
                if platform.system() == "Windows"
                else ctypes.CDLL(located)
            )
            return _HipRuntimeAlias(library)
        raise RuntimeError("HIP Runtime library (amdhip64) is unavailable")


class OpenGLFallbackBackend:
    """Own a hidden OpenGL context and validate a reusable texture/PBO ring."""

    def __init__(self, width: int, height: int, *, pbo_count: int = 3) -> None:
        if int(width) < 2 or int(height) < 2:
            raise ValueError("OpenGL fallback dimensions must be positive")
        if int(pbo_count) < 2:
            raise ValueError("OpenGL fallback requires at least two PBO slots")
        self.width = int(width)
        self.height = int(height)
        self.pbo_count = int(pbo_count)
        self._glfw = None
        self._gl = None
        self._glfw_initialized = False
        self._window = None
        self._texture = None
        self._framebuffer = None
        self._pbos: list[int] = []
        self._fences: list[Any] = []
        self._slot = 0
        self._previous_context = None
        self._closed = False
        self._cuda_interop: _CudaOpenGLInterop | None = None
        self._hip_interop: _HipOpenGLInterop | None = None
        self._rgba_staging = None
        self._rgba_staging_shape: tuple[int, int] | None = None
        self._rgba_staging_device = None
        try:
            self._capabilities = self._initialize()
        except Exception:
            self.close()
            raise

    @property
    def capabilities(self) -> OpenGLFallbackCapabilities:
        return self._capabilities

    @staticmethod
    def _cuda_tensor_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _initialize(self) -> OpenGLFallbackCapabilities:
        try:
            import glfw
            from OpenGL import GL
        except Exception as exc:
            raise RuntimeError(
                f"PyOpenGL/GLFW unavailable: {type(exc).__name__}: {exc}"
            ) from exc

        self._glfw = glfw
        self._gl = GL
        if not glfw.init():
            raise RuntimeError("glfwInit failed for OpenGL fallback")
        self._glfw_initialized = True

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if platform.system() == "Darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self._previous_context = glfw.get_current_context()
        self._window = glfw.create_window(
            self.width, self.height, "Desktop2Stereo OpenGL fallback", None, None
        )
        if not self._window:
            raise RuntimeError("GLFW could not create a hidden OpenGL context")
        glfw.make_context_current(self._window)

        texture = GL.glGenTextures(1)
        self._texture = int(texture[0] if isinstance(texture, (tuple, list)) else texture)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        generated = GL.glGenBuffers(self.pbo_count)
        if isinstance(generated, int):
            self._pbos = [generated]
        else:
            self._pbos = [int(value) for value in generated]
        if len(self._pbos) != self.pbo_count:
            raise RuntimeError("OpenGL returned an incomplete PBO ring")
        self._fences = [None] * self.pbo_count

        for pbo in self._pbos:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
            GL.glBufferData(
                GL.GL_PIXEL_UNPACK_BUFFER,
                self.width * self.height * 4,
                None,
                GL.GL_STREAM_DRAW,
            )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        framebuffer = GL.glGenFramebuffers(1)
        self._framebuffer = int(
            framebuffer[0] if isinstance(framebuffer, (tuple, list)) else framebuffer
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self._texture,
            0,
        )
        framebuffer_status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if framebuffer_status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"OpenGL RGBA8 framebuffer is incomplete: status={framebuffer_status}"
            )

        fence_supported = hasattr(GL, "glFenceSync") and hasattr(GL, "glClientWaitSync")
        if not fence_supported:
            raise RuntimeError("OpenGL sync objects are unavailable")
        version = GL.glGetString(GL.GL_VERSION)
        version_text = (
            version.decode("ascii", errors="replace")
            if isinstance(version, bytes)
            else str(version)
        )
        interop_details: list[str] = []
        force_host = os.environ.get("D2S_OPENGL_FORCE_HOST", "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if force_host:
            interop_details.append("forced host-upload probe")
            has_cuda = False
            has_hip = False
        else:
            try:
                import torch
                has_cuda = bool(torch.cuda.is_available())
                has_hip = bool(getattr(torch.version, "hip", None))
            except Exception:
                has_cuda = False
                has_hip = False
        if has_cuda and not has_hip and platform.system() != "Darwin":
            try:
                self._cuda_interop = _CudaOpenGLInterop(
                    texture=self._texture,
                    width=self.width,
                    height=self.height,
                    gl_target=GL.GL_TEXTURE_2D,
                )
            except Exception as exc:
                interop_details.append(
                    f"CUDA/OpenGL unavailable: {type(exc).__name__}: {exc}"
                )
        if has_hip and platform.system() != "Darwin" and self._cuda_interop is None:
            try:
                self._hip_interop = _HipOpenGLInterop(
                    texture=self._texture,
                    width=self.width,
                    height=self.height,
                    gl_target=GL.GL_TEXTURE_2D,
                )
            except Exception as exc:
                interop_details.append(
                    f"HIP/OpenGL unavailable: {type(exc).__name__}: {exc}"
                )
        if self._cuda_interop is not None:
            interop_details.append("CUDA/OpenGL interop active")
        if self._hip_interop is not None:
            interop_details.append("HIP/OpenGL interop active")
        if not interop_details:
            interop_details.append("CUDA/HIP unavailable")
        self._restore_context()
        gpu_interop = self._cuda_interop is not None or self._hip_interop is not None
        if not gpu_interop:
            interop_details.append("host-upload fallback; no portable OpenGL encoder interop")
        return OpenGLFallbackCapabilities(
            available=True,
            context_api=self._context_api(),
            texture_format="RGBA8",
            pbo_count=self.pbo_count,
            fence_supported=True,
            cuda_gl_interop=self._cuda_interop is not None,
            gpu_to_cpu=not gpu_interop,
            zero_copy=False,
            gpu_copy_count=2 if gpu_interop else 0,
            detail=(
                f"OpenGL {version_text}; framebuffer=complete; "
                f"{'; '.join(interop_details)}"
            ),
            framebuffer_supported=True,
            hip_gl_interop=self._hip_interop is not None,
            interop_mode=(
                "cuda"
                if self._cuda_interop is not None
                else "hip"
                if self._hip_interop is not None
                else "none"
            ),
        )

    def _context_api(self) -> str:
        system = platform.system()
        if system == "Windows":
            return "WGL"
        if system == "Darwin":
            return "NSGL"
        return "GLX/EGL"

    def _restore_context(self) -> None:
        if self._glfw is not None:
            self._glfw.make_context_current(self._previous_context)

    def _cuda_rgba_tensor(self, image: Any) -> Any:
        """Normalize a device frame into a reusable RGBA8 GPU staging tensor."""
        import torch

        image = getattr(image, "sbs", image)
        if not isinstance(image, torch.Tensor) or not bool(image.is_cuda):
            raise ValueError("OpenGL CUDA interop requires a CUDA tensor")
        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise ValueError(f"expected one frame, got {tuple(image.shape)!r}")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"unsupported CUDA SBS shape: {tuple(image.shape)!r}")
        if int(image.shape[0]) in (1, 3, 4):
            channels = int(image.shape[0])
            if channels == 1:
                image = image.expand(3, -1, -1)
            image = image[:3].permute(1, 2, 0)
        elif int(image.shape[-1]) in (1, 3, 4):
            channels = int(image.shape[-1])
            if channels == 1:
                image = image.expand(-1, -1, 3)
            image = image[..., :3]
        else:
            raise ValueError(f"unsupported CUDA SBS shape: {tuple(image.shape)!r}")
        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        image = image.contiguous()
        shape = (int(image.shape[0]), int(image.shape[1]))
        device = image.device
        if (
            self._rgba_staging is None
            or self._rgba_staging_shape != shape
            or self._rgba_staging_device != device
        ):
            self._rgba_staging = torch.empty(
                (shape[0], shape[1], 4), dtype=torch.uint8, device=device
            )
            self._rgba_staging_shape = shape
            self._rgba_staging_device = device
        self._rgba_staging[..., :3].copy_(image)
        self._rgba_staging[..., 3].fill_(255)
        return self._rgba_staging

    def submit_cuda_array(self, image: Any) -> int:
        """Upload one CUDA frame and return the persistent mapped CUarray."""
        if self._closed:
            raise RuntimeError("OpenGL fallback backend is closed")
        if self._cuda_interop is None:
            raise RuntimeError("CUDA/OpenGL interop is unavailable")
        rgba = self._cuda_rgba_tensor(image)
        previous = self._glfw.get_current_context()
        self._glfw.make_context_current(self._window)
        try:
            return self._cuda_interop.copy_to_array(rgba)
        finally:
            self._glfw.make_context_current(previous)

    def release_cuda_array(self) -> None:
        if self._cuda_interop is None or self._window is None:
            return
        previous = self._glfw.get_current_context()
        self._glfw.make_context_current(self._window)
        try:
            self._cuda_interop.release_persistent_array()
        finally:
            self._glfw.make_context_current(previous)

    def submit_cuda(self, image: Any) -> Any:
        if self._closed:
            raise RuntimeError("OpenGL fallback backend is closed")
        interop = self._cuda_interop or self._hip_interop
        if interop is None:
            raise RuntimeError("CUDA/HIP OpenGL interop is unavailable")
        rgba = self._cuda_rgba_tensor(image)
        previous = self._glfw.get_current_context()
        self._glfw.make_context_current(self._window)
        try:
            return interop.roundtrip(rgba)
        finally:
            self._glfw.make_context_current(previous)

    def submit_rgb(self, image: Any) -> np.ndarray:
        """Upload one RGB8 frame through the GL PBO/texture boundary."""
        if self._closed:
            raise RuntimeError("OpenGL fallback backend is closed")
        rgb = np.asarray(image)
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError(
                f"OpenGL fallback requires RGB8 {self.width}x{self.height}, "
                f"got shape={rgb.shape!r} dtype={rgb.dtype}"
            )
        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255

        glfw = self._glfw
        GL = self._gl
        previous = glfw.get_current_context()
        glfw.make_context_current(self._window)
        try:
            slot = self._slot
            pbo = self._pbos[slot]
            self._slot = (slot + 1) % len(self._pbos)
            previous_fence = self._fences[slot]
            if previous_fence is not None:
                status = GL.glClientWaitSync(
                    previous_fence,
                    GL.GL_SYNC_FLUSH_COMMANDS_BIT,
                    1_000_000_000,
                )
                GL.glDeleteSync(previous_fence)
                self._fences[slot] = None
                if status not in (
                    GL.GL_ALREADY_SIGNALED,
                    GL.GL_CONDITION_SATISFIED,
                ):
                    raise RuntimeError(
                        f"OpenGL PBO slot fence failed: status={status}"
                    )
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
            GL.glBufferSubData(GL.GL_PIXEL_UNPACK_BUFFER, 0, rgba)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                self.width,
                self.height,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                None,
            )
            self._fences[slot] = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            GL.glFlush()
        finally:
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            glfw.make_context_current(previous)
        return np.ascontiguousarray(rgb)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._glfw is None:
            return
        if self._window is None:
            if self._glfw_initialized:
                self._glfw.terminate()
                self._glfw_initialized = False
            return
        previous = self._glfw.get_current_context()
        self._glfw.make_context_current(self._window)
        try:
            if self._cuda_interop is not None:
                self._cuda_interop.close()
                self._cuda_interop = None
            if self._hip_interop is not None:
                self._hip_interop.close()
                self._hip_interop = None
            self._rgba_staging = None
            self._rgba_staging_shape = None
            self._rgba_staging_device = None
            if self._gl is not None:
                for fence in self._fences:
                    if fence is not None:
                        self._gl.glDeleteSync(fence)
                self._fences = []
                if self._pbos:
                    self._gl.glDeleteBuffers(len(self._pbos), self._pbos)
                if self._texture:
                    self._gl.glDeleteTextures(1, [self._texture])
                if self._framebuffer:
                    self._gl.glDeleteFramebuffers(1, [self._framebuffer])
                    self._framebuffer = None
        finally:
            self._glfw.destroy_window(self._window)
            self._glfw.make_context_current(self._previous_context or previous)
            self._window = None
            if self._glfw_initialized:
                self._glfw.terminate()
                self._glfw_initialized = False
