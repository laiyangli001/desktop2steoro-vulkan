"""OpenGL image fallback for the advanced streaming path.

This first implementation deliberately uses an OpenGL texture/PBO/fence
boundary and returns the CPU RGB frame to the existing FFmpeg host-upload
encoder. It is a reliable functional fallback, not a zero-copy encoder.
CUDA-OpenGL interop can replace the upload/readback boundary later without
changing the backend contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpenGLFallbackCapabilities:
    available: bool
    context_api: str
    texture_format: str
    pbo_count: int
    fence_supported: bool
    gpu_to_cpu: bool
    zero_copy: bool
    detail: str = ""


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
        self._window = None
        self._texture = None
        self._pbos: list[int] = []
        self._slot = 0
        self._previous_context = None
        self._closed = False
        self._capabilities = self._initialize()

    @property
    def capabilities(self) -> OpenGLFallbackCapabilities:
        return self._capabilities

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

        fence_supported = hasattr(GL, "glFenceSync") and hasattr(GL, "glClientWaitSync")
        if not fence_supported:
            raise RuntimeError("OpenGL sync objects are unavailable")
        version = GL.glGetString(GL.GL_VERSION)
        version_text = (
            version.decode("ascii", errors="replace")
            if isinstance(version, bytes)
            else str(version)
        )
        self._restore_context()
        return OpenGLFallbackCapabilities(
            available=True,
            context_api=self._context_api(),
            texture_format="RGBA8",
            pbo_count=self.pbo_count,
            fence_supported=True,
            gpu_to_cpu=True,
            zero_copy=False,
            detail=f"OpenGL {version_text}",
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

    def submit_rgb(self, image: Any) -> np.ndarray:
        """Upload one RGB8 frame through the GL PBO/texture boundary.

        The returned RGB buffer is intentionally passed to the existing
        host-upload FFmpeg encoder. This makes the fallback functional while
        exposing the exact current copy boundary in diagnostics.
        """
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
            pbo = self._pbos[self._slot]
            self._slot = (self._slot + 1) % len(self._pbos)
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
            sync = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            GL.glFlush()
            status = GL.glClientWaitSync(sync, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000)
            GL.glDeleteSync(sync)
            if status not in (
                GL.GL_ALREADY_SIGNALED,
                GL.GL_CONDITION_SATISFIED,
            ):
                raise RuntimeError(f"OpenGL texture upload fence failed: status={status}")
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        finally:
            glfw.make_context_current(previous)
        return np.ascontiguousarray(rgb)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._glfw is None or self._window is None:
            return
        previous = self._glfw.get_current_context()
        self._glfw.make_context_current(self._window)
        try:
            if self._gl is not None:
                if self._pbos:
                    self._gl.glDeleteBuffers(len(self._pbos), self._pbos)
                if self._texture:
                    self._gl.glDeleteTextures(1, [self._texture])
        finally:
            self._glfw.destroy_window(self._window)
            self._glfw.make_context_current(self._previous_context or previous)
            self._window = None
