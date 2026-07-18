from __future__ import annotations

import pytest

from stereo_runtime.motion_signal import RuntimeMotionSampler, clamp01


def torch_module():
    return pytest.importorskip("torch")


def test_clamp01_handles_numeric_and_invalid_values():
    assert clamp01(-1.0) == 0.0
    assert clamp01(0.25) == 0.25
    assert clamp01(2.0) == 1.0
    assert clamp01("bad") == 0.0


def test_runtime_motion_sampler_scores_cpu_tensor_motion():
    torch = torch_module()
    sampler = RuntimeMotionSampler()

    first = torch.zeros((3, 64, 64), dtype=torch.float32)
    second = torch.ones((3, 64, 64), dtype=torch.float32)

    assert sampler.sample(first) == 0.0
    assert sampler.sample(second) == 1.0


def test_runtime_motion_sampler_accepts_hwc_tensor():
    torch = torch_module()
    sampler = RuntimeMotionSampler()

    first = torch.zeros((64, 64, 3), dtype=torch.float32)
    second = torch.full((64, 64, 3), 0.25, dtype=torch.float32)

    assert sampler.sample(first) == 0.0
    assert sampler.sample(second) == 1.0


def test_runtime_motion_sampler_returns_last_score_for_invalid_shape():
    torch = torch_module()
    sampler = RuntimeMotionSampler()
    sampler.last_motion_score = 0.4

    assert sampler.sample(torch.zeros((64, 64), dtype=torch.float32)) == 0.4
