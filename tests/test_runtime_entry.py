from pathlib import Path


def test_openxr_starts_after_inference_load_and_first_ready_output() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/app_runtime/runtime_entry.py"
    ).read_text(encoding="utf-8")

    prepare = source.index("pipeline.prepare()")
    capture_start = source.index("capture_thread.start()")
    wait_ready = source.index(
        "_wait_for_runtime_ready(runtime_ready_event, pipeline_thread)"
    )
    presenter_create = source.index("presenter = OpenXrVulkanPresenter(")

    assert prepare < capture_start < wait_ready < presenter_create
