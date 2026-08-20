"""Optional Windows AMD AMF bridge loader.

The native bridge is intentionally optional. It only reports whether the AMD
AMF runtime and an AMD DXGI adapter are available; surface encoding is enabled
by the follow-up AMF/D3D11 frame path and never falls back silently to a CPU
"zero-copy" claim.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        root / "native" / "windows" / "d2s_amd_encoder.dll",
        root / "native" / "amd_encoder" / "d2s_amd_encoder.dll",
        Path(os.environ.get("D2S_AMD_ENCODER_DLL", "")),
    ]


def probe_amd_amf() -> tuple[bool, str]:
    """Return (available, diagnostic) without importing any GPU Python package."""

    if os.name != "nt":
        return False, "AMD AMF bridge is Windows-only"
    for candidate in _library_candidates():
        if not candidate or not candidate.exists():
            continue
        try:
            bridge = ctypes.WinDLL(str(candidate))
            bridge.d2s_amd_encoder_probe.restype = ctypes.c_int
            bridge.d2s_amd_encoder_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int]
            bridge.d2s_amd_encoder_last_error.restype = ctypes.c_int
            if bridge.d2s_amd_encoder_probe():
                return True, "AMD AMF runtime and DXGI adapter detected"
            buffer = ctypes.create_string_buffer(512)
            bridge.d2s_amd_encoder_last_error(buffer, len(buffer))
            return False, buffer.value.decode("utf-8", errors="replace")
        except OSError as exc:
            return False, f"AMD bridge load failed: {exc}"
    return False, "d2s_amd_encoder.dll is not installed"
