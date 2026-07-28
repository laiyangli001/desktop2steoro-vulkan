import pytest

from utils.screen_resolution_policy import (
    build_screen_sampling_plan,
    classify_input_resolution,
)


@pytest.mark.parametrize(
    ("width", "height", "tier"),
    (
        (1920, 1080, 1),
        (2560, 1440, 2),
        (3840, 2160, 4),
        (2560, 1600, 2),
        (3440, 1440, 4),
    ),
)
def test_input_resolution_uses_nearest_standard_tier(width, height, tier):
    assert classify_input_resolution(width, height) == tier


@pytest.mark.parametrize(
    ("source", "headset", "recommended", "effective", "filter_scale"),
    (
        ((1920, 1080), 2, 2, 2, 1.0),
        ((2560, 1440), 4, 4, 4, 1.0),
        ((3840, 2160), 8, 8, 8, 1.0),
        ((3840, 2160), 2, 8, 2, 2.0),
        ((1920, 1080), 8, 2, 2, 1.0),
    ),
)
def test_input_headset_matrix(source, headset, recommended, effective, filter_scale):
    plan = build_screen_sampling_plan(*source, headset)
    assert plan.recommended_headset_tier_k == recommended
    assert plan.effective_tier_k == effective
    assert plan.filter_scale == pytest.approx(filter_scale)


def test_invalid_resolution_is_rejected():
    with pytest.raises(ValueError):
        build_screen_sampling_plan(0, 1080, 2)
