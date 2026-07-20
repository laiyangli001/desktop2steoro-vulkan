from pathlib import Path

from tools.validate_shader_manifest import validate_manifest


def test_shader_manifest_matches_sources_and_spirv() -> None:
    root = Path(__file__).resolve().parents[1]
    assert validate_manifest(root) == []
