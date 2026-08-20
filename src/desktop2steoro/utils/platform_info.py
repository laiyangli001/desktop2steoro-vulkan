from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform

_GLIBC_DISTRO_TABLE = (
    ((2, 28), "debian10"),
    ((2, 31), "ubuntu20.04"),
    ((2, 35), "ubuntu22.04"),
    ((2, 36), "debian12"),
    ((2, 39), "ubuntu24.04"),
)


def get_os_name() -> str:
    return platform.system()


def get_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return machine


def get_linux_glibc_version() -> tuple[int, int]:
    if get_os_name() != "Linux":
        return (0, 0)

    try:
        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return (0, 0)
        libc = ctypes.CDLL(libc_name)
        gnu_get_libc_version = libc.gnu_get_libc_version
        gnu_get_libc_version.restype = ctypes.c_char_p
        version = gnu_get_libc_version().decode("ascii")
        major, minor, *_ = version.split(".")
        return (int(major), int(minor))
    except Exception:
        return (0, 0)


def get_linux_distro_id() -> str:
    override = os.environ.get("FLET_LINUX_DISTRO", "").strip()
    if override:
        return override

    glibc_version = get_linux_glibc_version()
    selected = None
    for required_glibc, distro_id in _GLIBC_DISTRO_TABLE:
        if glibc_version >= required_glibc:
            selected = distro_id
    return selected or _GLIBC_DISTRO_TABLE[0][1]


def get_flet_desktop_artifact_name(*, desktop_flavor: str | None = None) -> str | None:
    os_name = get_os_name()
    if os_name == "Windows":
        return "flet-windows.zip"
    if os_name == "Darwin":
        return "flet-macos.tar.gz"
    if os_name != "Linux":
        return None

    distro = get_linux_distro_id()
    arch = get_arch()
    flavor = _normalize_desktop_flavor(desktop_flavor)
    if flavor == "light":
        return f"flet-linux-{distro}-light-{arch}.tar.gz"
    return f"flet-linux-{distro}-{arch}.tar.gz"


def _normalize_desktop_flavor(desktop_flavor: str | None) -> str:
    if desktop_flavor is not None:
        flavor = desktop_flavor.strip().lower()
    else:
        flavor = os.environ.get("FLET_DESKTOP_FLAVOR", "").strip().lower()
    if flavor in ("full", "light"):
        return flavor
    return "light" if get_os_name() == "Linux" else "full"