from pathlib import Path

import pytest

from streaming.nvenc_cudaarray_bridge import (
    CudaTensorSurfaceView,
    NvencCudaArrayEncoder,
    _candidate_libraries,
)


def test_nvenc_cudaarray_binary_lives_in_feature_folder():
    candidates = _candidate_libraries()
    assert candidates
    assert candidates[0].name == "d2s_nvenc_cudaarray_bridge.dll"
    assert candidates[0].parent.name == "nvenc_cudaarray_bridge"


def test_nvenc_cudaarray_encoder_rejects_surface_switch():
    encoder = object.__new__(NvencCudaArrayEncoder)
    encoder._closed = False
    encoder._handle = object()
    encoder._cuda_array = 101
    with pytest.raises(ValueError, match="cannot switch CUDA arrays"):
        encoder.encode(202)


def test_nvenc_cudaarray_encoder_rejects_direct_tensor_surface_switch():
    encoder = object.__new__(NvencCudaArrayEncoder)
    encoder._closed = False
    encoder._handle = object()
    encoder._cuda_array = 101
    view = CudaTensorSurfaceView(
        cuda_array=202,
        device_pointer=303,
        channels=3,
        stride_y=192,
        stride_x=1,
        stride_c=2304,
        scalar_type=0,
        cuda_stream=404,
    )
    with pytest.raises(ValueError, match="cannot switch CUDA arrays"):
        encoder.encode(view)


def test_native_bridge_registers_cudaarray_and_abgr():
    root = Path(__file__).parents[1]
    source = (
        root / "native/nvenc_cudaarray_bridge/nvenc_cudaarray_bridge.cpp"
    ).read_text(encoding="utf-8")
    kernel = (
        root / "native/nvenc_cudaarray_bridge/cudaarray_surface_writer.cu"
    ).read_text(encoding="utf-8")
    workflow = (
        root / ".github/workflows/nvenc-cudaarray-bridge.yml"
    ).read_text(encoding="utf-8")

    assert "NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY" in source
    assert "NV_ENC_BUFFER_FORMAT_ABGR" in source
    assert "nvEncRegisterResource(CUDAARRAY)" in source
    assert "NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY" in source
    assert "d2s_nvenc_cudaarray_submit_tensor" in source
    assert "d2s_nvenc_cudaarray_read_packet_timed" in source
    assert "outputTimeStamp" in source
    assert "outputDuration" in source
    assert "surf2Dwrite" in kernel
    assert "cudaStreamSynchronize" in kernel
    assert "cudaMemcpy" not in kernel
    assert "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true" in workflow
    assert "Jimver/cuda-toolkit@v0.2.36" in workflow
    assert "actions/checkout@v5" in workflow
    assert "actions/upload-artifact@v6" in workflow
    assert "actions/download-artifact@v7" in workflow
    assert "src/desktop2stereo/streaming/nvenc_cudaarray_bridge" in workflow
