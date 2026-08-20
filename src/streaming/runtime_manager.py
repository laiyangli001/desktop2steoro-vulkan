from __future__ import annotations

import json
import os
import platform
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


def _runtime_key() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
    return f"{system}-{arch}"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                target = (destination / member.filename).resolve()
                if destination_resolved not in target.parents and target != destination_resolved:
                    raise ValueError(f"unsafe archive member: {member.filename}")
            bundle.extractall(destination)
        return
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)


def ensure_runtime(runtime_root: Path) -> tuple[Path, Path, Path]:
    manifest_path = runtime_root / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["runtimes"].get(_runtime_key())
    if entry is None:
        raise RuntimeError(f"no streaming runtime registered for {_runtime_key()}")
    ffmpeg = runtime_root / entry["ffmpeg_executable"]
    mediamtx = runtime_root / entry["mediamtx_executable"]
    config = runtime_root / entry.get("mediamtx_config", "mediamtx.yml")
    template = runtime_root / entry.get(
        "mediamtx_template", "mediamtx/mediamtx.yml"
    )
    if ffmpeg.is_file() and mediamtx.is_file() and template.is_file():
        if not config.is_file():
            config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template, config)
        if os.name != "nt":
            ffmpeg.chmod(ffmpeg.stat().st_mode | 0o111)
            mediamtx.chmod(mediamtx.stat().st_mode | 0o111)
        return ffmpeg, mediamtx, config
    archives = [runtime_root / entry["ffmpeg_archive"], runtime_root / entry["mediamtx_archive"]]
    missing = [str(path) for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing streaming runtime archive(s): " + ", ".join(missing))
    with tempfile.TemporaryDirectory(prefix="d2s-runtime-", dir=runtime_root) as temp_name:
        temp_root = Path(temp_name)
        ffmpeg_extract = temp_root / "ffmpeg"
        mediamtx_extract = temp_root / "mediamtx"
        ffmpeg_extract.mkdir()
        mediamtx_extract.mkdir()
        _safe_extract(archives[0], ffmpeg_extract)
        _safe_extract(archives[1], mediamtx_extract)
        ffmpeg_source = next(ffmpeg_extract.rglob(Path(entry["ffmpeg_executable"]).name))
        mediamtx_source = next(mediamtx_extract.rglob(Path(entry["mediamtx_executable"]).name))
        template_source = next(mediamtx_extract.rglob("mediamtx.yml"), None)
        ffmpeg.parent.mkdir(parents=True, exist_ok=True)
        mediamtx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ffmpeg_source, ffmpeg)
        shutil.copy2(mediamtx_source, mediamtx)
        if not template.is_file() and template_source is not None:
            template.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template_source, template)
    if not config.is_file() and template.is_file():
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template, config)
    if not config.is_file():
        raise FileNotFoundError(f"MediaMTX config template not found: {template}")
    if os.name != "nt":
        ffmpeg.chmod(ffmpeg.stat().st_mode | 0o111)
        mediamtx.chmod(mediamtx.stat().st_mode | 0o111)
    return ffmpeg, mediamtx, config
