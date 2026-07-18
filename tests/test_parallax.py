from __future__ import annotations

import pytest
import torch

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
from stereo_runtime.parallax import PARALLAX_BUDGET_TABLE, parallax_debug_info, resolve_parallax_budget


def test_resolve_parallax_budget_uses_short_side_table():
    budget = resolve_parallax_budget(
        render_width=1920,
        render_height=1080,
        preset="standard",
        convergence=0.0,
    )

    assert budget.max_disparity_px == 48.0
    assert budget.preset == "standard"
    assert budget.depth_response_name == "linear_clamp_convergence_v1"


def test_parallax_debug_info_records_depth_response_contract():
    budget = resolve_parallax_budget(
        render_width=1920,
        render_height=1080,
        preset="standard",
        convergence=0.0,
    )

    debug = parallax_debug_info(budget)

    assert debug["resolved_max_disparity_px"] == 48.0
    assert debug["parallax_budget_preset"] == "standard"
    assert debug["depth_response"] == "linear_clamp_convergence_v1"
    assert debug["parallax_resolver_version"] == 1


def test_resolve_parallax_budget_defaults_to_standard_preset():
    budget = resolve_parallax_budget(
        render_width=1920,
        render_height=1080,
        preset=None,
        convergence=0.0,
    )

    assert budget.max_disparity_px == 48.0
    assert budget.preset == "standard"


def test_resolve_parallax_budget_interpolates_between_resolution_levels():
    budget = resolve_parallax_budget(
        render_width=2560,
        render_height=1440,
        preset="standard",
        convergence=0.0,
    )

    assert budget.max_disparity_px == PARALLAX_BUDGET_TABLE["standard"][1440]


def test_resolve_parallax_budget_applies_ultrawide_aspect_protection():
    budget = resolve_parallax_budget(
        render_width=3840,
        render_height=1080,
        preset="standard",
        convergence=0.0,
    )

    assert budget.max_disparity_px == pytest.approx(48.0 * 0.70)


def test_compute_shift_px_uses_half_of_total_max_disparity_for_each_eye():
    depth = torch.ones(1, 1, 1, 1)
    shift = compute_shift_px(
        depth,
        1920,
        ShiftParams(depth_strength=1.0, convergence=0.0, max_disparity_px=96.0),
    )

    assert shift.item() == pytest.approx(-48.0)


def test_compute_shift_px_scales_actual_displacement_by_depth_strength():
    depth = torch.ones(1, 1, 1, 1)

    normal = compute_shift_px(depth, 1920, ShiftParams(depth_strength=1.0, convergence=0.0, max_disparity_px=40.0))
    strong = compute_shift_px(depth, 1920, ShiftParams(depth_strength=2.5, convergence=0.0, max_disparity_px=40.0))
    flat = compute_shift_px(depth, 1920, ShiftParams(depth_strength=0.0, convergence=0.0, max_disparity_px=40.0))

    assert normal.item() == pytest.approx(-20.0)
    assert strong.item() == pytest.approx(-50.0)
    assert flat.item() == pytest.approx(0.0)


def test_compute_shift_px_applies_layered_parallax_scales():
    depth = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    shift = compute_shift_px(
        depth,
        1920,
        ShiftParams(
            depth_strength=1.0,
            convergence=0.5,
            max_disparity_px=40.0,
            foreground_shift_scale=2.0,
            midground_shift_scale=1.0,
            background_shift_scale=0.5,
        ),
    )

    assert shift[0, 0, 0, 0].item() == pytest.approx(5.0)
    assert shift[0, 0, 0, 1].item() == pytest.approx(0.0)
    assert shift[0, 0, 0, 2].item() == pytest.approx(-20.0)


def test_shift_params_do_not_expose_legacy_ipd_formula_fields():
    fields = ShiftParams.__dataclass_fields__

    assert "ipd" not in fields
    assert "ipd_mm" not in fields
    assert "stereo_scale" not in fields
    assert "max_shift_ratio" not in fields

