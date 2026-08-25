from __future__ import annotations

import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest: dict, entry: dict, runtime_key: str) -> None:
    """Reject an unprovenanced runtime before it can be published."""
    if int(manifest.get("schema_version", 1)) < 2:
        return
    provenance = manifest.get("provenance")
    required_provenance = {
        "ffmpeg_source_repository",
        "ffmpeg_source_ref",
        "ffmpeg_build_id",
        "ffmpeg_configuration",
        "compiler",
    }
    if not isinstance(provenance, dict) or not required_provenance.issubset(provenance):
        missing = sorted(required_provenance - set(provenance or {}))
        raise RuntimeError(
            f"streaming runtime manifest provenance is incomplete for {runtime_key}: {missing}"
        )
    archive_hash = str(entry.get("ffmpeg_archive_sha256", "")).strip().casefold()
    if len(archive_hash) != 64 or any(char not in "0123456789abcdef" for char in archive_hash):
        raise RuntimeError(
            f"streaming runtime manifest has no valid FFmpeg archive SHA-256 for {runtime_key}"
        )


def _verify_archive(path: Path, expected_hash: str) -> None:
    actual_hash = _sha256(path)
    if actual_hash.casefold() != expected_hash.casefold():
        raise RuntimeError(
            f"streaming runtime archive SHA-256 mismatch: {path.name}; "
            f"expected={expected_hash} actual={actual_hash}"
        )


def ensure_runtime(runtime_root: Path) -> tuple[Path, Path, Path]:
    manifest_path = runtime_root / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_key = _runtime_key()
    entry = manifest["runtimes"].get(runtime_key)
    if entry is None:
        raise RuntimeError(f"no streaming runtime registered for {runtime_key}")
    _validate_manifest(manifest, entry, runtime_key)
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
    if not missing and int(manifest.get("schema_version", 1)) >= 2:
        _verify_archive(archives[0], entry["ffmpeg_archive_sha256"])
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
        ffmpeg_package_source = ffmpeg_source.parent.parent
        ffmpeg_package_destination = runtime_root / Path(entry["ffmpeg_executable"]).parts[0]
        if ffmpeg_package_destination.exists():
            shutil.rmtree(ffmpeg_package_destination)
        shutil.copytree(ffmpeg_package_source, ffmpeg_package_destination)
        mediamtx.parent.mkdir(parents=True, exist_ok=True)
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
