from __future__ import annotations

from utils import platform_info


def test_flet_artifact_for_windows(monkeypatch):
    monkeypatch.setattr(platform_info, "get_os_name", lambda: "Windows")

    assert platform_info.get_flet_desktop_artifact_name() == "flet-windows.zip"


def test_flet_artifact_for_macos(monkeypatch):
    monkeypatch.setattr(platform_info, "get_os_name", lambda: "Darwin")

    assert platform_info.get_flet_desktop_artifact_name() == "flet-macos.tar.gz"


def test_flet_artifact_for_linux_light_amd64(monkeypatch):
    monkeypatch.setattr(platform_info, "get_os_name", lambda: "Linux")
    monkeypatch.setattr(platform_info, "get_linux_distro_id", lambda: "ubuntu22.04")
    monkeypatch.setattr(platform_info, "get_arch", lambda: "amd64")
    monkeypatch.delenv("FLET_DESKTOP_FLAVOR", raising=False)

    assert (
        platform_info.get_flet_desktop_artifact_name()
        == "flet-linux-ubuntu22.04-light-amd64.tar.gz"
    )


def test_flet_artifact_for_linux_full_arm64(monkeypatch):
    monkeypatch.setattr(platform_info, "get_os_name", lambda: "Linux")
    monkeypatch.setattr(platform_info, "get_linux_distro_id", lambda: "debian12")
    monkeypatch.setattr(platform_info, "get_arch", lambda: "arm64")

    assert (
        platform_info.get_flet_desktop_artifact_name(desktop_flavor="full")
        == "flet-linux-debian12-arm64.tar.gz"
    )


def test_linux_distro_override(monkeypatch):
    monkeypatch.setenv("FLET_LINUX_DISTRO", "ubuntu22.04")

    assert platform_info.get_linux_distro_id() == "ubuntu22.04"


def test_linux_distro_from_glibc(monkeypatch):
    monkeypatch.delenv("FLET_LINUX_DISTRO", raising=False)
    monkeypatch.setattr(platform_info, "get_linux_glibc_version", lambda: (2, 36))

    assert platform_info.get_linux_distro_id() == "debian12"


def test_arch_normalization(monkeypatch):
    monkeypatch.setattr(platform_info.platform, "machine", lambda: "x86_64")
    assert platform_info.get_arch() == "amd64"

    monkeypatch.setattr(platform_info.platform, "machine", lambda: "aarch64")
    assert platform_info.get_arch() == "arm64"