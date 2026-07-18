from __future__ import annotations

from types import SimpleNamespace

from stereo_runtime.session_helpers import StereoRuntimeLogger, StereoWarmupTracker


class FakeRuntime:
    def __init__(self):
        self.stereo_config = SimpleNamespace(
            backend="fast_plus",
            layers=4,
            hole_fill="none",
            edge_dilation=2,
            output_format="half_sbs",
            temporal=True,
        )
        self.config = SimpleNamespace(
            output_format="half_sbs",
            stereo_quality="fast_plus",
            stereo_preset="cinema",
            mode="cinema",
        )
        self.warmup_calls = 0

    def warmup_stereo_kernels_for_frame(self, rgb_frame):
        self.warmup_calls += 1


class FakeFrame:
    shape = (3, 32, 32)
    dtype = "float32"
    device = "cpu"


def test_stereo_warmup_tracker_deduplicates_by_frame_key():
    runtime = FakeRuntime()
    tracker = StereoWarmupTracker(runtime, run_mode="Viewer", openxr_runtime_direct=False)

    tracker.warmup_once_for_frame(FakeFrame())
    tracker.warmup_once_for_frame(FakeFrame())

    assert runtime.warmup_calls == 1
    assert tracker.key_for_frame(FakeFrame())[0] == (3, 32, 32)


def test_stereo_warmup_tracker_skips_openxr_direct_without_full_synthesis_preset():
    runtime = FakeRuntime()
    tracker = StereoWarmupTracker(
        runtime,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        active_preset="traditional_fastest",
    )

    tracker.warmup_once_for_frame(FakeFrame())

    assert runtime.warmup_calls == 0


def test_stereo_warmup_tracker_runs_for_openxr_full_synthesis_preset():
    runtime = FakeRuntime()
    tracker = StereoWarmupTracker(
        runtime,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        active_preset="cinema",
    )

    tracker.warmup_once_for_frame(FakeFrame())

    assert runtime.warmup_calls == 1


def test_stereo_runtime_logger_deduplicates_mode_logs(capsys, monkeypatch):
    monkeypatch.setenv("D2S_DEBUG", "1")
    runtime = FakeRuntime()
    logger = StereoRuntimeLogger(runtime, active_preset_getter=lambda: "cinema")

    logger.log_mode_once()
    logger.log_mode_once()

    output = capsys.readouterr().out
    assert output.count("[Main] Stereo mode active:") == 1
    assert "preset=cinema" in output


def test_stereo_runtime_logger_deduplicates_fused_logs(capsys, monkeypatch):
    monkeypatch.setenv("D2S_DEBUG", "1")
    runtime = FakeRuntime()
    logger = StereoRuntimeLogger(runtime, active_preset_getter=lambda: "cinema")
    result = SimpleNamespace(
        debug_info={
            "backend": "fast_plus",
            "runtime_output_format": "half_sbs",
            "runtime_output_dtype": "uint8",
            "runtime_output_pack_backend": "torch",
            "fast_plus_fused_backend": "triton",
            "fast_plus_fused_skip": "none",
            "fast_plus_fused_temporal_bypass": "false",
        }
    )

    logger.log_fast_plus_fused_runtime_state(result)
    logger.log_fast_plus_fused_runtime_state(result)

    output = capsys.readouterr().out
    assert output.count("[Main] Stereo runtime output:") == 1
    assert "fast_plus_fused=triton" in output

def test_stereo_runtime_logger_openxr_output_log(capsys, monkeypatch):
    monkeypatch.setenv("D2S_DEBUG", "1")
    runtime = FakeRuntime()
    logger = StereoRuntimeLogger(runtime, active_preset_getter=lambda: "cinema")
    result = SimpleNamespace(
        shader_uniforms={
            "max_disparity_px": 96.0,
            "convergence": 0.12,
            "depth_response": "linear_clamp_convergence_v1",
        },
        debug_info={
            "backend": "openxr_roll_adaptive_grid_sample",
            "runtime_output_format": "openxr_eye_views",
            "runtime_output_dtype": "uint8",
            "runtime_output_eye_size": "3840x2160",
            "resolved_max_disparity_px": 9.0,
            "openxr_convergence": 9.0,
            "depth_response": "debug_fallback",
        },
    )

    logger.log_fast_plus_fused_runtime_state(result)
    logger.log_fast_plus_fused_runtime_state(result)

    output = capsys.readouterr().out
    assert output.count("[Main] Stereo runtime output:") == 1
    assert "output=openxr_eye_views" in output
    assert "dtype=uint8" in output
    assert "eye=3840x2160" in output
    assert "max_disparity=96.000" in output
    assert "convergence=0.120" in output
    assert "depth_response=linear_clamp_convergence_v1" in output
    assert "max_disparity=9.000" not in output
    assert "depth_response=debug_fallback" not in output
    assert "fast_plus_fused" not in output
    assert "pack=" not in output


def test_stereo_runtime_logger_prefers_structured_output_fields(capsys, monkeypatch):
    monkeypatch.setenv("D2S_DEBUG", "1")
    runtime = FakeRuntime()
    logger = StereoRuntimeLogger(runtime, active_preset_getter=lambda: "cinema")
    result = SimpleNamespace(
        output_format="openxr_eye_views",
        output_dtype="uint8",
        output_eye_size=(3840, 2160),
        debug_info={
            "backend": "openxr_roll_adaptive_grid_sample",
            "runtime_output_format": "legacy_format",
            "runtime_output_dtype": "legacy_dtype",
            "runtime_output_eye_size": "1x1",
        },
    )

    logger.log_fast_plus_fused_runtime_state(result)

    output = capsys.readouterr().out
    assert "output=openxr_eye_views" in output
    assert "dtype=uint8" in output
    assert "eye=3840x2160" in output
