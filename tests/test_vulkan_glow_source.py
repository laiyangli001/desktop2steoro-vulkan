from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from stereo_runtime.vulkan_glow_source import VulkanGlowSourceComputeBackend
from stereo_runtime.vulkan_glow_source_pass import VulkanGlowSourcePass


def test_glow_prefilter_uses_legacy_mip_footprint_in_source_pixels() -> None:
    assert VulkanGlowSourceComputeBackend.prefilter_scale("veil", 5.4) == 1.0
    assert VulkanGlowSourceComputeBackend.prefilter_scale("frosted", 5.0) == 32.0
    assert VulkanGlowSourceComputeBackend.prefilter_scale("glow", 5.4) == 256.0
    assert VulkanGlowSourceComputeBackend.prefilter_scale("surround", 5.4) == 256.0


def test_glow_pass_contract_is_fixed_rgba_target() -> None:
    effect_pass = object.__new__(VulkanGlowSourcePass)
    effect_pass.target_width = 320
    effect_pass.target_height = 180

    assert effect_pass.group_counts == (40, 23, 1)
    assert effect_pass.input_buffer_size(3840, 2160) == 3840 * 2160 * 12


def test_glow_fence_poll_handles_pyvulkan_not_ready_exception() -> None:
    class VkNotReady(Exception):
        pass

    class Vk:
        def __init__(self) -> None:
            self.ready = False

        def vkGetFenceStatus(self, _device, _fence):
            if not self.ready:
                raise VkNotReady()
            return None

    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend.vk = Vk()
    backend.vk.VkNotReady = VkNotReady
    backend.context = SimpleNamespace(device=object())

    assert backend._fence_complete(object()) is False
    backend.vk.ready = True
    assert backend._fence_complete(object()) is True


def test_glow_frame_lease_is_counted_once_and_released() -> None:
    slot = SimpleNamespace(
        image=SimpleNamespace(
            resource=SimpleNamespace(width=320, height=180, format=37)
        ),
        lease_count=0,
    )
    backend = object.__new__(VulkanGlowSourceComputeBackend)
    backend._closed = False
    backend._current_slot = slot
    backend._frame_slots = {}
    backend._serial = 3
    backend._last_submit_ms = 0.2
    backend._reuse_count = 0
    backend._budget_skip_count = 0
    backend.poll = lambda: None

    first = backend.acquire(8)
    second = backend.acquire(8)
    assert first["glow_vulkan_image"] is slot.image.resource
    assert second["glow_vulkan_serial"] == 3
    assert slot.lease_count == 1

    backend.release_frame(8)
    assert slot.lease_count == 0


def test_glow_shader_and_spirv_are_checked_in_together() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "shaders/d2s_glow_source.comp").read_text(encoding="utf-8")
    spirv = (root / "shaders/d2s_glow_source.spv").read_bytes()

    assert "max(base_footprint, vec2(max(params.prefilter_scale, 1.0)))" in source
    assert "srgb_to_linear" in source
    assert "vec2(output_uv.x, 1.0 - output_uv.y)" in source
    assert spirv[:4] == b"\x03\x02#\x07"
