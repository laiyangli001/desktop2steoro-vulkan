from pathlib import Path

from tools.validate_shader_manifest import validate_manifest


def test_shader_manifest_matches_sources_and_spirv() -> None:
    root = Path(__file__).resolve().parents[1]
    assert validate_manifest(root) == []


def test_runtime_shader_assets_stay_inside_src_product_boundary() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "shaders").exists()
    assert (root / "src" / "shaders" / "manifest.json").is_file()
    assert list((root / "src" / "shaders").glob("*.spv"))
