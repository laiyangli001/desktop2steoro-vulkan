from __future__ import annotations

import zipfile

from gui import flet_runtime


def _write_client_archive(path, content: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("flet/flet.exe", content)


def test_vendored_flet_cache_tracks_archive_content(tmp_path, monkeypatch) -> None:
    packages_dir = tmp_path / "packages"
    clients_dir = tmp_path / "clients"
    packages_dir.mkdir()
    archive_path = packages_dir / "flet-windows.zip"
    _write_client_archive(archive_path, "0.85.3")

    monkeypatch.setattr(flet_runtime, "PACKAGES_DIR", packages_dir)
    monkeypatch.setattr(flet_runtime, "CLIENTS_DIR", clients_dir)
    monkeypatch.setattr(flet_runtime, "_current_artifact_name", lambda: archive_path.name)
    monkeypatch.setattr(flet_runtime, "_is_linux", lambda: False)
    monkeypatch.setattr(flet_runtime, "get_os_name", lambda: "Windows")

    view_path = flet_runtime.ensure_vendored_flet_view()
    executable = clients_dir / "flet-windows" / "flet" / "flet.exe"
    assert view_path == str(executable.parent)
    assert executable.read_text() == "0.85.3"

    _write_client_archive(archive_path, "0.86.5")
    flet_runtime.ensure_vendored_flet_view()

    assert executable.read_text() == "0.86.5"
