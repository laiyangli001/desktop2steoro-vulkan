"""Non-reversible device identity used for server-side license binding."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import uuid
from pathlib import Path


def device_hash() -> str:
    material = "|".join(("d2s-device-v1", platform.system(), _platform_identity(), str(uuid.getnode())))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _platform_identity() -> str:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
                if value:
                    return str(value)
        except (OSError, ImportError):
            pass
    elif system == "Linux":
        values = []
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id", "/sys/class/dmi/id/product_uuid"):
            try:
                value = Path(path).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                values.append(value)
        if values:
            return "|".join(values)
    elif system == "Darwin":
        try:
            result = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True, timeout=2, check=False)
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split("=", 1)[-1].strip().strip('"')
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.node() or os.name
