from pathlib import Path

from path_config import APP_ROOT

from gui.config import DEFAULTS
from gui.localization import (
    display_to_parallel_inference_workers,
    get_messages,
    parallel_inference_options,
    parallel_inference_to_display,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"


def test_parallel_inference_defaults_to_single_worker() -> None:
    assert DEFAULTS["Parallel Inference"] is False
    assert DEFAULTS["Parallel Inference Workers"] == 1


def test_parallel_inference_is_left_of_cross_eyed_in_advanced_row() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    start = source.index("stereo_row4 = ft.Row([")
    end = source.index("], spacing=1)", start)
    row_source = source[start:end]

    assert row_source.index("self.parallel_inference_dd") < row_source.index(
        "self.cross_eyed_cb"
    )
    assert "stereo_row4" in source[source.index("self._advanced_stereo_rows"):]


def test_parallel_inference_tooltip_recommends_two_stage_pipeline() -> None:
    tooltip = get_messages("ZH")["tooltip_parallel_inference"]
    assert "推荐两路" in tooltip
    assert "深度推理与 SBS 合成流水并行" in tooltip
    assert "三路" in tooltip


def test_parallel_inference_options_are_localized_without_changing_worker_count() -> None:
    assert parallel_inference_options("EN") == [
        "Single Inference", "Dual Inference", "Triple Inference"
    ]
    assert parallel_inference_options("CN") == ["单路推理", "两路推理", "三路推理"]

    for workers in (1, 2, 3):
        assert display_to_parallel_inference_workers(
            parallel_inference_to_display(workers, "EN")
        ) == workers
        assert display_to_parallel_inference_workers(
            parallel_inference_to_display(workers, "CN")
        ) == workers


def test_controller_model_options_come_only_from_scanned_directories() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    assert "options=ctrl_dirs" in source
    assert 'options=["None", *ctrl_dirs]' not in source
    assert 'value="PICO"' in source
