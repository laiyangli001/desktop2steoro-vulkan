from __future__ import annotations

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
    assert "gpu_to_cpu=False gpu_copy=True zero_copy=False" in source
    assert "std::unordered_map<VkImage, ConvertSlot> convert_slots" in source
    assert "encoder->convert_slots.try_emplace(command_key)" in source
    assert "slot.descriptor_set" in source
