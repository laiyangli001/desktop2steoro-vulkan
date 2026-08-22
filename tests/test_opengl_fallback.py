from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from streaming.opengl_stream_backend import OpenGLFallbackBackend, _HipRuntimeAlias


def test_opengl_backend_requires_two_pbo_slots():
    with pytest.raises(ValueError, match="at least two PBO"):
        OpenGLFallbackBackend(64, 36, pbo_count=1)


def test_opengl_backend_rejects_wrong_rgb_frame_without_context():
    backend = object.__new__(OpenGLFallbackBackend)
    backend.width = 64
    backend.height = 36
    backend._closed = False
    with pytest.raises(ValueError, match="requires RGB8"):
        backend.submit_rgb(np.zeros((36, 64, 4), dtype=np.uint8))


def test_opengl_backend_creates_texture_pbo_fence_and_submits_frame():
    try:
        backend = OpenGLFallbackBackend(64, 36)
    except RuntimeError as exc:
        pytest.skip(f"OpenGL context unavailable in test environment: {exc}")

    try:
        assert backend.capabilities.available is True
        assert backend.capabilities.texture_format == "RGBA8"
        assert backend.capabilities.pbo_count == 3
        assert backend.capabilities.fence_supported is True
        assert backend.capabilities.zero_copy is False

        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[0, 0] = [7, 23, 42]
        returned = backend.submit_rgb(frame)

        assert returned.shape == frame.shape
        assert returned.dtype == np.uint8
        assert returned.flags.c_contiguous
        np.testing.assert_array_equal(returned, frame)
    finally:
        backend.close()


def test_opengl_cuda_interop_roundtrip_stays_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        backend = OpenGLFallbackBackend(64, 36)
    except RuntimeError as exc:
        pytest.skip(f"OpenGL context unavailable in test environment: {exc}")
    try:
        if not backend.capabilities.cuda_gl_interop:
            pytest.skip(backend.capabilities.detail)
        source = torch.zeros((3, 36, 64), device="cuda", dtype=torch.uint8)
        source[0, 0, 0] = 19
        roundtrip = backend.submit_cuda(source)
        assert bool(roundtrip.is_cuda)
        assert tuple(roundtrip.shape) == (36, 64, 4)
        assert int(roundtrip[0, 0, 0].item()) == 19
        assert int(roundtrip[0, 0, 3].item()) == 255
    finally:
        backend.close()


def test_vulkan_failure_selects_cuda_opengl_nvenc_branch(monkeypatch):
    import streaming.direct_sbs as direct_sbs

    class FakeTensor:
        is_cuda = True
        ndim = 3
        shape = (3, 36, 64)

    class FakeBackend:
        capabilities = SimpleNamespace(
            context_api="WGL",
            texture_format="RGBA8",
            pbo_count=3,
            fence_supported=True,
            cuda_gl_interop=True,
            gpu_to_cpu=False,
            zero_copy=False,
        )

        def submit_cuda(self, frame):
            assert frame is fake_frame
            return "cuda-rgba"

        def close(self):
            pass

    class FakePacketOutput:
        def submit_cuda_frame(self, frame):
            assert frame == "cuda-rgba"

    class FakePyNv:
        def __init__(self):
            self._pynv_output = FakePacketOutput()
            self.server_process = None

        def _start_ffmpeg(self, width, height):
            assert (width, height) == (64, 36)

        def close(self):
            pass

    fake_frame = FakeTensor()
    output = object.__new__(direct_sbs.VulkanDirectSbsOutput)
    output._opengl_fallback_attempted = False
    output._opengl_fallback = None
    output._opengl_pynv_fallback = None
    output._opengl_fallback_active = False
    output._host_fallback = None
    output.fps = 30
    output.server_process = None
    output._new_opengl_pynv_fallback = FakePyNv
    monkeypatch.setattr(direct_sbs, "OpenGLFallbackBackend", lambda *args, **kwargs: FakeBackend())

    output._fallback_to_opengl(fake_frame, RuntimeError("vulkan probe failed"))

    assert output._opengl_fallback_active is True
    assert output._opengl_pynv_fallback is not None
    assert output._host_fallback is None


def test_hip_runtime_alias_maps_graphics_api_symbols():
    class FakeHipRuntime:
        hipGraphicsGLRegisterImage = object()
        hipGetErrorString = object()

    alias = _HipRuntimeAlias(FakeHipRuntime())
    assert alias.cudaGraphicsGLRegisterImage is FakeHipRuntime.hipGraphicsGLRegisterImage
    assert alias.cudaGetErrorString is FakeHipRuntime.hipGetErrorString


def test_opengl_fallback_contract_is_documented_and_wired():
    source = (Path(__file__).parents[1] / "src/desktop2steoro/streaming/direct_sbs.py").read_text(
        encoding="utf-8"
    )
    backend_source = (
        Path(__file__).parents[1]
        / "src/desktop2steoro/streaming/opengl_stream_backend.py"
    ).read_text(encoding="utf-8")

    assert "OpenGLFallbackBackend" in source
    assert "_fallback_to_opengl" in source
    assert "D2S_STATUS] Vulkan unavailable; using OpenGL fallback" in source
    assert "glGenBuffers" in backend_source
    assert "glFenceSync" in backend_source
    assert "cudaGraphicsGLRegisterImage" in backend_source
    assert "cudaMemcpy2DFromArrayAsync" in backend_source
    assert "_HipOpenGLInterop" in backend_source
    assert 'name = "hip" + name[4:]' in backend_source
    assert "encoder=AMF" in source
    assert "interop={getattr(caps, 'interop_mode', 'none')}" in source
    assert "fallback.submit_frame(rgb)" in source
    assert "zero_copy=False" in backend_source


def test_vulkan_doc_records_current_opengl_copy_boundary():
    document = (
        Path(__file__).parents[1]
        / "docs/16-advanced-streaming-vulkan-image-path.md"
    ).read_text(encoding="utf-8")
    assert "OpenGL" in document
    assert "gpu_to_cpu=True" in document
    assert "interop=none" in document
    assert "zero_copy=False" in document
