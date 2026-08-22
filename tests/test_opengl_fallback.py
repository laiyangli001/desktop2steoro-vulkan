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
        assert backend.capabilities.framebuffer_supported is True
        assert backend.capabilities.pbo_count == 3
        assert backend.capabilities.fence_supported is True
        assert backend.capabilities.zero_copy is False
        assert backend.capabilities.gpu_copy_count == (
            2
            if backend.capabilities.cuda_gl_interop or backend.capabilities.hip_gl_interop
            else 0
        )

        frame = np.zeros((36, 64, 3), dtype=np.uint8)
        frame[0, 0] = [7, 23, 42]
        returned = backend.submit_rgb(frame)
        for index in range(5):
            frame[0, 0] = [7 + index, 23, 42]
            returned = backend.submit_rgb(frame)

        assert len(backend._fences) == 3
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
        staging = backend._rgba_staging
        assert staging is not None
        second = backend.submit_cuda(source)
        assert backend._rgba_staging is staging
        assert bool(roundtrip.is_cuda)
        assert bool(second.is_cuda)
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


def test_no_interop_uses_host_encoder_without_cpu_gl_roundtrip(monkeypatch):
    import streaming.direct_sbs as direct_sbs

    class FakeFrame:
        sbs = object()

    class FakeBackend:
        capabilities = SimpleNamespace(
            context_api="NSGL",
            texture_format="RGBA8",
            pbo_count=3,
            fence_supported=True,
            cuda_gl_interop=False,
            hip_gl_interop=False,
            interop_mode="none",
            gpu_to_cpu=True,
            zero_copy=False,
        )

        def submit_rgb(self, _frame):
            raise AssertionError("host fallback must not upload RGB to OpenGL")

        def close(self):
            pass

    class FakeHost:
        video_encoder = "h264_videotoolbox"
        server_process = None

        def __init__(self):
            self.frames = []

        def submit_frame(self, frame):
            self.frames.append(frame)

        def close(self):
            pass

    rgb = np.zeros((36, 64, 3), dtype=np.uint8)
    host = FakeHost()
    output = object.__new__(direct_sbs.VulkanDirectSbsOutput)
    output._opengl_fallback_attempted = False
    output._opengl_fallback = None
    output._opengl_pynv_fallback = None
    output._opengl_amd_fallback = None
    output._opengl_fallback_active = False
    output._host_fallback = None
    output.fps = 30
    output.server_process = None
    output._new_host_fallback = lambda: host
    monkeypatch.setattr(direct_sbs, "OpenGLFallbackBackend", lambda *args, **kwargs: FakeBackend())
    monkeypatch.setattr(direct_sbs, "runtime_sbs_to_rgb", lambda _frame: rgb)

    output.submit_frame(rgb)
    output.submit_frame(rgb)

    assert output._opengl_fallback_active is True
    assert output._host_fallback is host
    assert len(host.frames) == 2
    for submitted in host.frames:
        np.testing.assert_array_equal(submitted, rgb)


def test_opengl_runtime_submit_failure_switches_to_stable_host_path(monkeypatch):
    import streaming.direct_sbs as direct_sbs

    class FakeFrame:
        sbs = object()

    class FakeBackend:
        capabilities = SimpleNamespace(
            context_api="WGL",
            texture_format="RGBA8",
            pbo_count=3,
            fence_supported=True,
            cuda_gl_interop=True,
            hip_gl_interop=False,
            interop_mode="cuda",
            gpu_to_cpu=False,
            zero_copy=False,
        )
        closed = False

        def submit_cuda(self, _frame):
            raise RuntimeError("OpenGL interop lost")

        def close(self):
            self.closed = True

    class FakeHost:
        video_encoder = "h264_nvenc"
        server_process = None

        def __init__(self):
            self.frames = []

        def submit_frame(self, frame):
            self.frames.append(frame)

        def close(self):
            pass

    backend = FakeBackend()
    host = FakeHost()
    rgb = np.zeros((36, 64, 3), dtype=np.uint8)
    output = object.__new__(direct_sbs.VulkanDirectSbsOutput)
    output._opengl_fallback_attempted = True
    output._opengl_fallback = backend
    output._opengl_pynv_fallback = None
    output._opengl_amd_fallback = None
    output._opengl_fallback_active = True
    output._host_fallback = None
    output.server_process = None
    output._new_host_fallback = lambda: host
    output._stop_native = lambda: None
    monkeypatch.setattr(direct_sbs, "runtime_sbs_to_rgb", lambda _frame: rgb)

    output.submit_cuda_frame(FakeFrame())

    assert backend.closed is True
    assert output._opengl_fallback_active is False
    assert output._host_fallback is host
    assert host.frames == [rgb]


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
    assert "glGenFramebuffers" in backend_source
    assert "glCheckFramebufferStatus" in backend_source
    assert "glFenceSync" in backend_source
    assert "self._fences" in backend_source
    assert "PBO slot fence failed" in backend_source
    assert "host-upload fallback; no portable OpenGL encoder interop" in backend_source
    assert "self._glfw_initialized = False" in backend_source
    assert "except Exception:\n            self.close()\n            raise" in backend_source
    assert "cudaGraphicsGLRegisterImage" in backend_source
    assert "cudaMemcpy2DFromArrayAsync" in backend_source
    assert "_HipOpenGLInterop" in backend_source
    assert 'name = "hip" + name[4:]' in backend_source
    assert "encoder=AMF" in source
    assert "interop={getattr(caps, 'interop_mode', 'none')}" in source
    assert "fallback.submit_frame(rgb)" in source
    assert "zero_copy=False" in backend_source
    assert "D2S_OPENGL_FORCE_HOST" in backend_source
    assert "framebuffer_supported=True" in backend_source
    assert "gpu_copy_count=2 if gpu_interop else 0" in backend_source
    assert "gpu_copy_count" in source


def test_opengl_smoke_tool_contract_is_present():
    tool = (
        Path(__file__).parents[1]
        / "src/desktop2steoro/tools/opengl_fallback_smoke.py"
    ).read_text(encoding="utf-8")
    assert "OpenGLFallbackBackend" in tool
    assert "--require-gpu-interop" in tool
    assert "--force-host" in tool
    assert "gpu_probe" in tool
    assert "host_probe" in tool
    assert '"path": "gpu-interop"' in tool
    assert '"path": "host-upload"' in tool
    assert "json.dumps" in tool
    assert "_SOURCE_ROOT = Path(__file__).resolve().parents[1]" in tool
    assert "sys.path.insert(0, str(_SOURCE_ROOT))" in tool


def test_opengl_rtsp_soak_tool_forces_real_fallback_boundary():
    tool = (
        Path(__file__).parents[1]
        / "src/desktop2steoro/tools/opengl_fallback_rtsp_soak.py"
    ).read_text(encoding="utf-8")
    assert "VulkanDirectSbsOutput" in tool
    assert "output._native_vulkan_bridge = None" in tool
    assert "output.submit_cuda_frame(tensor)" in tool
    assert "output._opengl_fallback_active" in tool
    assert "cuda-opengl-interop" in tool
    assert "--force-host" in tool
    assert "--cpu" in tool
    assert "output.submit_frame(tensor)" in tool
    assert "D2S_OPENGL_FORCE_HOST" in tool
    assert "opengl_fallback_rtsp_soak: PASS" in tool


def test_vulkan_doc_records_current_opengl_copy_boundary():
    document = (
        Path(__file__).parents[1]
        / "docs/16-advanced-streaming-vulkan-image-path.md"
    ).read_text(encoding="utf-8")
    assert "OpenGL" in document
    assert "gpu_to_cpu=True" in document
    assert "interop=none" in document
    assert "host-upload fallback" in document
    assert "zero_copy=False" in document
