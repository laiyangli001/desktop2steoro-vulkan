from __future__ import annotations

from pathlib import Path

import pytest

from path_config import APP_ROOT
from streaming.vulkan_bridge import VulkanNativeBridge


class _FakeLibrary:
    def d2s_vulkan_ffmpeg_bridge_abi_version(self):
        return 5

    class _Probe:
        argtypes = None
        restype = None

        def __call__(self, output, capacity):
            output.value = b"ok"
            return 1

    d2s_vulkan_ffmpeg_bridge_probe = _Probe()
    d2s_vulkan_ffmpeg_encoder_create = staticmethod(lambda *args: None)
    d2s_vulkan_ffmpeg_encoder_acquire_frame = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_acquire_rgba_frame = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_release_rgba_frame = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_encode_rgba_frame = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_submit_frame = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_submit_image = staticmethod(lambda *args: None)
    d2s_vulkan_ffmpeg_encoder_read_packet = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_flush = staticmethod(lambda *args: 0)
    d2s_vulkan_ffmpeg_encoder_destroy = staticmethod(lambda *args: None)


def test_native_bridge_accepts_only_versioned_contract() -> None:
    bridge = VulkanNativeBridge(_FakeLibrary())
    assert bridge.library is not None


def test_native_bridge_missing_library_is_explicit(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        VulkanNativeBridge.load(tmp_path / "missing.dll")


def test_native_bridge_has_platform_feature_directory_candidate(monkeypatch) -> None:
    monkeypatch.setattr("streaming.vulkan_bridge.platform.system", lambda: "Windows")

    candidate = VulkanNativeBridge._bundled_candidate()

    assert candidate == Path(__file__).resolve().parents[1] / (
        "src/desktop2stereo/streaming/vulkan_ffmpeg_bridge/windows/"
        "d2s_vulkan_ffmpeg_bridge.dll"
    )


def test_native_bridge_declares_low_latency_gpu_path_and_diagnostic_log():
    source = (
        APP_ROOT.parents[1]
        / "native"
        / "vulkan_ffmpeg_bridge"
        / "bridge.cpp"
    ).read_text(encoding="utf-8")

    assert "result->codec->max_b_frames = 0" in source
    assert "input=RGBA8 encode=NV12" in source
    assert "queue_prepare=%u queue_compute=%u bf=0" in source
    assert "gpu_to_cpu=False gpu_copy=%s zero_copy=%s" in source
    assert "supports_direct_nv12_storage" in source
    assert "create_plane_storage_view" in source
    assert "VK_IMAGE_USAGE_STORAGE_BIT" in source
    assert "const VkPipelineStageFlags2 output_stage = direct_storage" in source
    assert "waits[1].stageMask = direct_storage" in source
    assert "std::unordered_map<VkImage, ConvertSlot> convert_slots" in source
    assert "encoder->convert_slots.try_emplace(command_key)" in source
    assert "slot.descriptor_set" in source
    assert "d2s_vulkan_ffmpeg_encoder_device_identity" in source
    assert "vkGetPhysicalDeviceProperties2" in source
    header = (
        APP_ROOT.parents[1]
        / "native"
        / "vulkan_ffmpeg_bridge"
        / "vulkan_ffmpeg_bridge.h"
    ).read_text(encoding="utf-8")
    assert "d2s_vulkan_ffmpeg_encoder_device_identity" in header
    assert "result == AVERROR(EAGAIN) || result == AVERROR_EOF" in source


def test_vulkan_bridge_workflow_uses_shared_ffmpeg_runtime() -> None:
    workflow = (
        APP_ROOT.parents[1] / ".github/workflows/vulkan-ffmpeg-bridge.yml"
    ).read_text(encoding="utf-8")

    assert "D2S_SHARED_FFMPEG_RUNTIME" in workflow
    assert "Shared FFmpeg runtime dependency version mismatch" in workflow
    assert "shared_linux=src/desktop2stereo/streaming/rtmp/ffmpeg/lib" in workflow
    assert "! -name d2s_vulkan_ffmpeg_bridge.dll -delete" in workflow
    assert "! -name d2s_vulkan_ffmpeg_bridge.so -delete" in workflow
    assert "cp artifact/windows/*" not in workflow
