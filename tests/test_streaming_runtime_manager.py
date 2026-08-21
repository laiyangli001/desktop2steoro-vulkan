from __future__ import annotations

import json
import zipfile
from pathlib import Path

from streaming import runtime_manager


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_ensure_runtime_installs_complete_ffmpeg_package(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = {
        "schema_version": 1,
        "runtimes": {
            "Windows-amd64": {
                "ffmpeg_archive": "ffmpeg.zip",
                "mediamtx_archive": "mediamtx.zip",
                "ffmpeg_executable": "ffmpeg/bin/ffmpeg.exe",
                "mediamtx_executable": "mediamtx/mediamtx.exe",
                "mediamtx_config": "mediamtx.yml",
                "mediamtx_template": "mediamtx/mediamtx.yml",
            }
        },
    }
    (tmp_path / "runtime-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_zip(
        tmp_path / "ffmpeg.zip",
        {
            "ffmpeg/bin/ffmpeg.exe": b"ffmpeg",
            "ffmpeg/bin/avcodec.dll": b"codec",
            "ffmpeg/lib/avcodec.lib": b"import-library",
        },
    )
    _write_zip(
        tmp_path / "mediamtx.zip",
        {
            "mediamtx.exe": b"mediamtx",
            "mediamtx.yml": b"paths: {}",
        },
    )
    monkeypatch.setattr(runtime_manager, "_runtime_key", lambda: "Windows-amd64")

    ffmpeg, mediamtx, config = runtime_manager.ensure_runtime(tmp_path)

    assert ffmpeg.read_bytes() == b"ffmpeg"
    assert (tmp_path / "ffmpeg/bin/avcodec.dll").read_bytes() == b"codec"
    assert (tmp_path / "ffmpeg/lib/avcodec.lib").read_bytes() == b"import-library"
    assert mediamtx.read_bytes() == b"mediamtx"
    assert config.read_bytes() == b"paths: {}"
