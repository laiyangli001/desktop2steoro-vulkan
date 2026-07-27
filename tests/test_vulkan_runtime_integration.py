from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

import stereo_runtime.runtime as runtime_module
from stereo_runtime import StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult
from stereo_runtime.openxr_render import OpenXRRenderConfig


class _Provider:
    def load(self):
        return None

    def predict_profile(self, rgb):
        b, _, height, width = rgb.shape
        depth = torch.linspace(0.0, 1.0, width, dtype=torch.float32).view(1, 1, 1, width)
        depth = depth.expand(b, 1, height, width).contiguous()
        return DepthProfileResult(depth=depth, preprocess_ms=0.0, model_ms=0.0, postprocess_ms=0.0)

    def close(self):
        return None


class _FakeVulkanBackend:
    def submit_frame(self, rgb, depth, *, params):
        del params
        mask = torch.zeros_like(depth)
        return rgb, rgb.clone(), mask, {
            "vulkan_device": "fake-vulkan",
            "vulkan_fused_backend": "vulkan_stereo_fused",
            "vulkan_readback": "host_visible_storage_buffer",
        }

    def submit_layered_frame(self, rgb, depth, *, params):
        del params
        mask = torch.zeros_like(depth)
        return rgb, rgb.clone(), mask, {
            "vulkan_device": "fake-vulkan",
            "vulkan_fused_backend": "vulkan_stereo_layered",
            "vulkan_readback": "host_visible_storage_buffer",
        }

    def close(self):
        return None


def test_openxr_prewarp_is_enabled_by_default_for_presenter_owned_output(monkeypatch):
    monkeypatch.delenv("D2S_OPENXR_PREWARP_EYES", raising=False)

    assert runtime_module._openxr_prewarp_eyes_enabled() is True


def test_presenter_owned_vulkan_compute_declares_cuda_external_input_path():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/stereo_runtime/vulkan_backend.py"
    ).read_text(encoding="utf-8")
    assert "VulkanExportableBuffer" in source
    assert "copy_tensor_to_buffer" in source
    assert 'input_mode = "cuda_external_buffer"' in source
    assert 'input_mode = "host_visible_buffer"' in source
    assert "wait_semaphore=wait_semaphore" in source


def test_vulkan_output_shader_decodes_srgb_before_unorm_store():
    shader = (
        Path(__file__).resolve().parents[1]
        / "shaders"
        / "d2s_stereo_layered_output.comp"
    ).read_text(encoding="utf-8")

    assert "display-referred sRGB" in shader
    assert "pow((encoded + 0.055) / 1.055" in shader
    assert "output image is an UNORM storage image" in shader
    assert "float mask = screen_edge_suppressed" not in shader
    assert "if (found >= 1.0) return 1.0;" in shader


def test_vulkan_openxr_output_shader_uses_legacy_openxr_eye_order():
    shader = (
        Path(__file__).resolve().parents[1]
        / "shaders"
        / "d2s_stereo_layered_output.comp"
    ).read_text(encoding="utf-8")

    assert "fill_eye(ix, iy, -1.0, fill_mask)" in shader
    assert "params.symmetric != 0u ? 1.0 : 0.9" in shader


def test_vulkan_generic_sbs_shader_keeps_synthesis_eye_order():
    shader = (
        Path(__file__).resolve().parents[1]
        / "shaders"
        / "d2s_stereo_layered.comp"
    ).read_text(encoding="utf-8")

    assert "fill_eye(ix, iy, 1.0" in shader
    assert "params.symmetric != 0u ? -1.0 : -0.9" in shader


def test_vulkan_tiled_reference_shader_keeps_layered_pass_abi():
    shader = (
        Path(__file__).resolve().parents[1]
        / "shaders"
        / "d2s_stereo_layered_tiled.comp"
    ).read_text(encoding="utf-8")
    backend_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stereo_runtime"
        / "vulkan_backend.py"
    ).read_text(encoding="utf-8")

    assert "layout(local_size_x = 16, local_size_y = 16" in shader
    assert "shared float shared_depth" in shader
    assert "layout(set = 0, binding = 4, std430) writeonly buffer MaskBuffer" in shader
    assert "layered_shader_path" in backend_source
    assert "d2s_stereo_layered.spv" in backend_source


def test_vulkan_fallback_openxr_output_normalizes_generic_eye_order():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stereo_runtime"
        / "runtime.py"
    ).read_text(encoding="utf-8")

    assert "left_eye = vulkan_stereo.right_eye" in source
    assert "right_eye = vulkan_stereo.left_eye" in source


def test_openxr_depth_temporal_is_disabled_with_stereo_temporal(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    first = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    second = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    assert torch.equal(
        runtime._stabilize_openxr_rgb_depth(first, enabled=False), first
    )
    assert torch.equal(
        runtime._stabilize_openxr_rgb_depth(second, enabled=False), second
    )
    runtime.close()


def test_openxr_default_auto_mode_builds_deferred_vulkan_request(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=True, vendor="nvidia"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        temporal=False,
        foreground_shift_scale=1.15,
        midground_shift_scale=1.05,
        background_shift_scale=1.05,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(),
    )

    assert result.vulkan_compute_request is not None
    assert result.debug_info["vulkan_zero_copy_deferred"] == 1
    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.debug_info["vulkan_zero_copy_reason"] == "ready"
    assert result.vulkan_compute_request.params.foreground_scale == 1.15
    assert result.vulkan_compute_request.params.midground_scale == 1.05
    assert result.vulkan_compute_request.params.background_scale == 1.05
    runtime.close()


def test_runtime_routes_fast_plus_to_vulkan_fused_backend(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unknown"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="vulkan",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.debug_info["vulkan_fused_backend"] == "vulkan_stereo_fused"
    assert result.debug_info["vulkan_device"] == "fake-vulkan"
    assert result.left_eye.shape == (1, 3, 8, 12)
    assert result.right_eye.shape == (1, 3, 8, 12)
    runtime.close()


def test_runtime_auto_routes_to_vulkan_when_triton_probe_fails(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="auto",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert runtime._resolved_stereo_compute_backend == "vulkan"
    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    runtime.close()


def test_openxr_prewarp_path_prefers_deferred_request_after_vulkan_backend_init(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="fast_plus",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert result.vulkan_compute_request is not None
    assert result.debug_info["vulkan_zero_copy_request"] == 1
    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.output_format == "openxr_eye_views"
    runtime.close()


def test_openxr_prewarp_vulkan_defers_to_presenter_zero_copy_request(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)

    result = runtime.process_openxr_frame(
        torch.rand(1, 3, 8, 12),
        OpenXRRenderConfig(output_mode="full_synthesis_eyes"),
    )

    assert result.vulkan_compute_request is not None
    assert result.debug_info["vulkan_zero_copy_request"] == 1
    assert result.vulkan_compute_request.rgb.shape == (1, 3, 8, 12)
    assert result.vulkan_compute_request.depth.shape == (1, 1, 8, 12)
    runtime.close()


def test_runtime_routes_quality_4k_to_vulkan_layered_backend(monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "probe_triton_runtime",
        lambda device=None: SimpleNamespace(available=False, vendor="unsupported"),
    )
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        stereo_quality="quality_4k",
        stereo_compute_backend="vulkan",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=_Provider(), collect_memory_stats=False)
    runtime._vulkan_stereo_backend = _FakeVulkanBackend()

    result = runtime.process_rgb_frame(torch.rand(1, 3, 8, 12))

    assert result.debug_info["stereo_compute_backend"] == "vulkan"
    assert result.debug_info["vulkan_fused_backend"] == "vulkan_stereo_layered"
    assert result.debug_info["sbs_backend"] == "vulkan_layered_stereo"
    runtime.close()
