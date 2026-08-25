from pathlib import Path

import pytest

from streaming.native_rtsp_output import (
    NativeRtspAvOutput,
    _opus_library_candidates,
    _pack_rtp_header,
)
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


def test_native_opus_binary_lives_in_feature_folder():
    candidates = _opus_library_candidates()
    packaged = Path(candidates[1])
    assert packaged.name == "opus.dll"
    assert packaged.parent.name == "native_rtsp_output"
    assert packaged.is_absolute()


def test_native_rtp_headers_use_announced_payload_types():
    h264 = _pack_rtp_header(
        payload_type=96, marker=True, sequence=7, timestamp=9000, ssrc=11
    )
    opus = _pack_rtp_header(
        payload_type=111, marker=False, sequence=8, timestamp=960, ssrc=12
    )
    assert h264[0] == 0x80
    assert h264[1] == 0x80 | 96
    assert opus[0] == 0x80
    assert opus[1] == 111


def test_native_h264_rtp_is_batched_without_losing_packet_boundaries():
    class RecordingSocket:
        def __init__(self):
            self.calls = []

        def sendall(self, data):
            self.calls.append(bytes(data))

    output = NativeRtspAvOutput(
        object(), host="127.0.0.1", port=8554, stream_key="live", fps=25
    )
    recording_socket = RecordingSocket()
    output._socket = recording_socket
    output._send_h264(b"\x00\x00\x00\x01\x65" + b"x" * 200_000, 9000)

    assert 1 < len(recording_socket.calls) < 10
    packet_count = 0
    for batch in recording_socket.calls:
        cursor = 0
        while cursor < len(batch):
            assert batch[cursor] == ord("$")
            assert batch[cursor + 1] == 0
            size = int.from_bytes(batch[cursor + 2 : cursor + 4], "big")
            packet = batch[cursor + 4 : cursor + 4 + size]
            assert len(packet) == size
            assert packet[1] & 0x7F == 96
            packet_count += 1
            cursor += 4 + size
        assert cursor == len(batch)
    assert packet_count > len(recording_socket.calls)


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
