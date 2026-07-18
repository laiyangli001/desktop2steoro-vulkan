from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


_OPTIONAL_MODULES = {
    "flet": "flet",
    "torch": "torch",
    "vulkan": "vulkan",
    "openxr": "xr",
    "tensorrt": "tensorrt",
    "windows_capture_cuda": "wc_cuda",
    "windows_capture_rocm": "wc_rocm",
}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _probe_vulkan_device() -> dict[str, object]:
    try:
        from viewer.vulkan_context import VulkanContext

        with VulkanContext.create() as context:
            device = context.device_info
            return {
                "available": True,
                "device": device.name,
                "api_version": device.api_version_text,
                "graphics_queue_family": device.queue_family_index,
            }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _probe_openxr_extensions() -> dict[str, object]:
    try:
        import xr

        extensions = {
            item.extension_name.decode("utf-8")
            if isinstance(item.extension_name, bytes)
            else str(item.extension_name)
            for item in xr.enumerate_instance_extension_properties()
        }
        return {
            "loader_available": True,
            "vulkan_enable2": "XR_KHR_vulkan_enable2" in extensions,
        }
    except Exception as exc:
        return {
            "loader_available": False,
            "vulkan_enable2": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_capability_report() -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    filament_names = {
        "win32": "filament_bridge.dll",
        "darwin": "libfilament_bridge.dylib",
        "linux": "libfilament_bridge.so",
    }
    filament_name = filament_names.get(sys.platform)
    filament_path = src_root / "xr_viewer" / "native" / filament_name if filament_name else None
    return {
        "project": "desktop2steoro-vulkan",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_modules": {
            name: _module_available(module_name) for name, module_name in _OPTIONAL_MODULES.items()
        },
        "vulkan": _probe_vulkan_device(),
        "openxr": _probe_openxr_extensions(),
        "filament_bridge": {
            "expected_path": str(filament_path) if filament_path else None,
            "available": bool(filament_path and filament_path.is_file()),
        },
        "migration": {
            "python_vulkan_runtime": "phase1_implemented",
            "openxr_vulkan_session": "phase1_validated",
            "filament_vulkan_bridge": "pending",
        },
    }

