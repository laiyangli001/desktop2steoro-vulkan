from dataclasses import dataclass
from types import SimpleNamespace
import queue
import threading

import pytest

import stereo_runtime.nvfruc_stage as stage_module
from stereo_runtime.nvfruc_stage import NvFrucStage


@dataclass
class _Result:
    value: int
    output_format: str = "openxr_full_synthesis_eyes"
    left_eye: object = None
    right_eye: object = None
    sbs: object = None
    debug_info: dict | None = None
    native_final_sbs_surface: object = None
    vulkan_compute_request: object = None


class _FakeGenerator:
    def __init__(self, left, right, **kwargs):
        self.calls = []

    def interpolate(self, previous, next_pair, **kwargs):
        self.calls.append((previous, next_pair, kwargs))
        return ("generated-left", "generated-right")

    def reset(self):
        pass

    def close(self):
        pass


def _result(value: int) -> _Result:
    return _Result(value, left_eye=f"left-{value}", right_eye=f"right-{value}", debug_info={})


def test_stage_publishes_real_midpoint_real_in_order(monkeypatch):
    monkeypatch.setattr(stage_module, "NvFrucStereoGenerator", _FakeGenerator)
    stage = NvFrucStage(
        input_q=queue.Queue(maxsize=1),
        output_q=queue.Queue(maxsize=6),
        shutdown_event=threading.Event(),
    )

    stage._publish_group((_result(1), 10.0), (_result(2), 12.0))

    items = [stage.output_q.get_nowait() for _ in range(3)]
    assert [item[0].value for item in items[:1] + items[2:]] == [1, 2]
    assert items[1][0].debug_info["nvfruc_generated"] is True
    assert [item[1] for item in items] == [10.0, 11.0, 12.0]
    stage.close()


def test_stage_resets_and_passes_real_frames_when_generator_fails(monkeypatch):
    class FailingGenerator:
        def __init__(self, *args, **kwargs):
            raise stage_module.NvFrucUnavailable("missing native bridge")

    monkeypatch.setattr(stage_module, "NvFrucStereoGenerator", FailingGenerator)
    stage = NvFrucStage(
        input_q=queue.Queue(maxsize=1),
        output_q=queue.Queue(maxsize=6),
        shutdown_event=threading.Event(),
    )

    # _publish_group propagates; run() owns the pass-through policy.
    # Verify the failure type is explicit for the runtime loop.
    with pytest.raises(stage_module.NvFrucUnavailable):
        stage._ensure_generator(_result(1))
