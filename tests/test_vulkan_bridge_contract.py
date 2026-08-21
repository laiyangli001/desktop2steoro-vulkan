from __future__ import annotations

import pytest

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
