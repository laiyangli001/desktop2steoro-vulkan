from pathlib import Path

import numpy as np
import pytest

from streaming.opengl_stream_backend import OpenGLFallbackBackend


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
        assert backend.capabilities.gpu_to_cpu is True
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
    assert "zero_copy=False" in backend_source


def test_vulkan_doc_records_current_opengl_copy_boundary():
    document = (
        Path(__file__).parents[1]
        / "docs/16-advanced-streaming-vulkan-image-path.md"
    ).read_text(encoding="utf-8")
    assert "OpenGL" in document
    assert "gpu_to_cpu=True" in document
    assert "zero_copy=False" in document
