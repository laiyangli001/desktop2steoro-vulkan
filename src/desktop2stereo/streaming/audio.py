from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Iterable


STEREO_MIX_NAMES = [
    # English
    "stereo mix",
    "what you hear",
    "loopback",
    "system audio",
    "wave out mix",
    "mixed output",
    # Chinese
    "立体声混音",
    "您听到的声音",
    "环路",
    "系统音频",
    "波形输出混合",
    "混合输出",
    # Japanese
    "ステレオ ミックス",
    "ステレオミックス",
    "ループバック",
    "システムオーディオ",
    "ミックス出力",
    # Spanish
    "mezcla estéreo",
    "lo que escuchas",
    "bucle",
    "audio del sistema",
    "salida mixta",
    # French
    "mixage stéréo",
    "bouclage",
    "audio système",
    "sortie mixte",
    # German
    "stereomix",
    "was du hörst",
    "loopback",
    "systemaudio",
    "gemischte ausgabe",
    # macOS specific
    "blackhole",
    "aggregate device",
    "multi-output device",
    "virtual desktop speakers",
    "remote sound",
    # Linux specific
    "monitor",
]


_DSHOW_AUDIO_RE = re.compile(
    # FFmpeg 6 and earlier use ``[dshow @ ...]``; newer builds may
    # prefix the same device lines with ``[in#0 ...]``.
    r'^\[[^\]]+\]\s+"([^"]+)"\s+\(audio\)\s*',
    re.MULTILINE,
)


_WASAPI_AUDIO_RE = re.compile(
    r'^\[[^\]]+\]\s+"([^"]+)"\s+\(audio\)\s*',
    re.MULTILINE,
)


def parse_ffmpeg_dshow_audio_devices(output: str) -> list[str]:
    devices = []
    seen = set()
    for match in _DSHOW_AUDIO_RE.finditer(str(output or "")):
        name = match.group(1).strip()
        key = name.casefold()
        if name and key not in seen:
            devices.append(name)
            seen.add(key)
    return devices


def parse_ffmpeg_wasapi_audio_devices(output: str) -> list[str]:
    devices = []
    seen = set()
    for match in _WASAPI_AUDIO_RE.finditer(str(output or "")):
        name = match.group(1).strip()
        key = name.casefold()
        if name and key not in seen:
            devices.append(name)
            seen.add(key)
    return devices


def query_ffmpeg_wasapi_audio_devices(
    ffmpeg_path: str | Path,
) -> list[str] | None:
    executable = Path(ffmpeg_path)
    if not executable.is_file():
        return None
    try:
        result = subprocess.run(
            [
                str(executable), "-nostdin", "-hide_banner",
                "-list_devices", "true", "-f", "wasapi", "-i", "dummy",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = "\n".join((result.stdout or "", result.stderr or ""))
    devices = parse_ffmpeg_wasapi_audio_devices(output)
    return devices or None


def query_ffmpeg_dshow_audio_devices(
    ffmpeg_path: str | Path,
) -> list[str] | None:
    executable = Path(ffmpeg_path)
    if not executable.is_file():
        return None
    try:
        result = subprocess.run(
            [
                str(executable),
                "-nostdin",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = "\n".join((result.stdout or "", result.stderr or ""))
    devices = parse_ffmpeg_dshow_audio_devices(output)
    if not devices:
        return None
    return devices


def find_loopback_audio_devices(
    device_names: Iterable[object],
) -> list[str]:
    devices = []
    seen = set()
    for value in device_names:
        name = str(value or "").strip()
        normalized = name.casefold()
        if not name or normalized in seen:
            continue
        is_stereo_mix = (
            "audio stereo input" not in normalized
            and any(token in normalized for token in STEREO_MIX_NAMES)
        )
        if is_stereo_mix or "virtual-audio-capturer" in normalized:
            devices.append(name)
            seen.add(normalized)
    return devices
