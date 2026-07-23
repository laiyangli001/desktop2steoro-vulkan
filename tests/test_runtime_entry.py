import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ENTRY = ROOT / "src/app_runtime/runtime_entry.py"


def _load_environment_resolver():
    source = RUNTIME_ENTRY.read_text(encoding="utf-8")
    module = ast.parse(source)
    resolver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_filament_environment_paths"
    )
    namespace = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(RUNTIME_ENTRY), "exec"), namespace)
    return namespace["_resolve_filament_environment_paths"]


def test_openxr_starts_after_inference_load_and_first_ready_output() -> None:
    source = RUNTIME_ENTRY.read_text(encoding="utf-8")

    prepare = source.index("pipeline.prepare()")
    capture_start = source.index("capture_thread.start()")
    wait_ready = source.index(
        "_wait_for_runtime_ready(runtime_ready_event, pipeline_thread)"
    )
    presenter_create = source.index("presenter = OpenXrVulkanPresenter(")

    assert prepare < capture_start < wait_ready < presenter_create


def test_openxr_environment_uses_selected_folder_and_profile_glb(tmp_path: Path) -> None:
    resolver = _load_environment_resolver()
    room = tmp_path / "xr_viewer/environments/3D_Artemis"
    room.mkdir(parents=True)
    (room / "profile.json").write_text(
        json.dumps({"glb": "environment-custom.glb"}),
        encoding="utf-8",
    )
    (room / "environment-custom.glb").write_bytes(b"glTF")

    glb_path, profile_path = resolver(
        {"Environment Model": "3D_Artemis"},
        tmp_path,
    )

    assert glb_path == room / "environment-custom.glb"
    assert profile_path == room / "profile.json"


def test_openxr_environment_missing_profile_falls_back_to_default(
    tmp_path: Path,
) -> None:
    resolver = _load_environment_resolver()
    default = tmp_path / "xr_viewer/environments/Default"
    default.mkdir(parents=True)
    (default / "profile.json").write_text(
        json.dumps({"glb": None}),
        encoding="utf-8",
    )

    glb_path, profile_path = resolver(
        {"Environment Model": "MissingRoom"},
        tmp_path,
    )

    assert glb_path is None
    assert profile_path == default / "profile.json"


def test_openxr_environment_without_selection_uses_default(tmp_path: Path) -> None:
    resolver = _load_environment_resolver()
    default = tmp_path / "xr_viewer/environments/Default"
    default.mkdir(parents=True)
    (default / "profile.json").write_text(
        json.dumps({"glb": None}),
        encoding="utf-8",
    )

    glb_path, profile_path = resolver({}, tmp_path)

    assert glb_path is None
    assert profile_path == default / "profile.json"
