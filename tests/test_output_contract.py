from types import SimpleNamespace

import pytest

from app_runtime.output_contract import LatestFrameOutputRouter, VulkanStereoOutputFrame


def _frame(frame_id: int = 1) -> VulkanStereoOutputFrame:
    return VulkanStereoOutputFrame(
        frame_id=frame_id,
        timestamp=2.0,
        left_eye=SimpleNamespace(name="left"),
        right_eye=SimpleNamespace(name="right"),
        sbs=SimpleNamespace(name="sbs"),
        ready_timeline=4,
    )


def test_output_router_fans_out_one_latest_gpu_frame() -> None:
    received = []
    router = LatestFrameOutputRouter()
    router.add_sink("openxr", SimpleNamespace(submit=received.append))
    router.add_sink("preview", SimpleNamespace(submit=received.append))

    router.publish(_frame(1))
    router.publish(_frame(2))

    assert router.latest.frame_id == 2
    assert [frame.frame_id for frame in received] == [1, 1, 2, 2]
    assert router.sink_names == ("openxr", "preview")


def test_output_frame_and_router_reject_invalid_state() -> None:
    with pytest.raises(ValueError, match="both left_eye"):
        VulkanStereoOutputFrame(1, 0.0, None, object())
    router = LatestFrameOutputRouter()
    router.close()
    with pytest.raises(RuntimeError, match="closed"):
        router.publish(_frame())


def test_output_frame_rejects_unknown_color_space() -> None:
    with pytest.raises(ValueError, match="color_space"):
        VulkanStereoOutputFrame(1, 0.0, object(), object(), color_space="pq")
