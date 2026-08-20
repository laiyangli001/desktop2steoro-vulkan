from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from project_paths import load_project_paths


MEDIAMTX_API = "https://api.github.com/repos/bluenviron/mediamtx/releases/latest"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Desktop2Stereo-runtime"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        return
    with tarfile.open(archive, "r:*") as bundle:
        bundle.extractall(destination)


def _asset_url(system: str, machine: str) -> tuple[str, str]:
    if system == "Windows":
        if machine in {"AMD64", "x86_64"}:
            return "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip", "zip"
        raise ValueError(f"unsupported Windows architecture: {machine}")
    if system == "Linux":
        if machine in {"x86_64", "AMD64"}:
            return "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz", "tar.xz"
        if machine in {"aarch64", "arm64"}:
            return "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz", "tar.xz"
        raise ValueError(f"unsupported Linux architecture: {machine}")
    if system == "Darwin":
        if machine in {"arm64", "aarch64"}:
            return "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip", "zip"
        if machine in {"x86_64", "AMD64"}:
            return "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip", "zip"
        raise ValueError(f"unsupported macOS architecture: {machine}")
    raise ValueError(f"unsupported operating system: {system}")


def _mediamtx_asset(system: str, machine: str, assets: list[dict]) -> str:
    os_name = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}[system]
    arch = {"AMD64": "amd64", "x86_64": "amd64", "arm64": "arm64", "aarch64": "arm64"}[machine]
    suffix = ".zip" if system == "Windows" else ".tar.gz"
    needle = f"{os_name}_{arch}{suffix}"
    for asset in assets:
        if asset["name"].endswith(needle):
            return asset["browser_download_url"]
    raise ValueError(f"MediaMTX has no release asset for {os_name}/{arch}")


def _find_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"archive did not contain {name}")
    return matches[0]


def install(runtime_root: Path, system: str, machine: str) -> None:
    paths = load_project_paths()
    ffmpeg_url, ffmpeg_format = _asset_url(system, machine)
    mediamtx_release = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(MEDIAMTX_API, headers={"User-Agent": "Desktop2Stereo-runtime"}),
            timeout=30,
        ).read().decode("utf-8")
    )
    mediamtx_url = _mediamtx_asset(system, machine, mediamtx_release["assets"])
    ffmpeg_dir = runtime_root / "ffmpeg" / "bin"
    mediamtx_dir = runtime_root / "mediamtx"
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d2s-streaming-") as temp:
        temp_root = Path(temp)
        ffmpeg_archive = temp_root / f"ffmpeg.{ffmpeg_format.replace('.', '')}"
        mediamtx_archive = temp_root / ("mediamtx.zip" if mediamtx_url.endswith(".zip") else "mediamtx.tar.gz")
        _download(ffmpeg_url, ffmpeg_archive)
        _download(mediamtx_url, mediamtx_archive)
        ffmpeg_extract = temp_root / "ffmpeg-extract"
        mediamtx_extract = temp_root / "mediamtx-extract"
        ffmpeg_extract.mkdir()
        mediamtx_extract.mkdir()
        _extract_archive(ffmpeg_archive, ffmpeg_extract)
        _extract_archive(mediamtx_archive, mediamtx_extract)
        ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        mediamtx_dir.mkdir(parents=True, exist_ok=True)
        source_ffmpeg = _find_file(ffmpeg_extract, "ffmpeg.exe" if system == "Windows" else "ffmpeg")
        source_mediamtx = _find_file(mediamtx_extract, "mediamtx.exe" if system == "Windows" else "mediamtx")
        shutil.copy2(source_ffmpeg, ffmpeg_dir / source_ffmpeg.name)
        shutil.copy2(source_mediamtx, mediamtx_dir / source_mediamtx.name)
    template_config = runtime_root / "mediamtx" / "mediamtx.yml"
    project_config = runtime_root / "mediamtx.yml"
    bundled_config = paths.app_dir / "streaming" / "rtmp" / "mediamtx" / "mediamtx.yml"
    if not template_config.exists() and bundled_config.exists():
        template_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled_config, template_config)
    if not project_config.exists() and template_config.exists():
        shutil.copy2(template_config, project_config)
    print(f"Installed streaming runtime under {runtime_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download FFmpeg and MediaMTX for Desktop2Stereo")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--system", choices=("Windows", "Linux", "Darwin"), default=platform.system())
    parser.add_argument("--machine", default=platform.machine())
    args = parser.parse_args()
    paths = load_project_paths()
    output = args.output or (paths.app_dir / "streaming" / "rtmp")
    install(output, args.system, args.machine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
