from __future__ import annotations

from tools import intel_native_hardware_smoke


def test_hardware_smoke_rejects_non_windows_without_loading_native_bridges(monkeypatch):
    monkeypatch.setattr(intel_native_hardware_smoke.os, "name", "posix")

    assert intel_native_hardware_smoke.main(["--model", "missing.xml"]) == 1
