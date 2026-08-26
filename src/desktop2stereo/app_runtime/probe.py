from __future__ import annotations

import importlib.util
import os
import platform
import sys
from pathlib import Path


_OPTIONAL_MODULES = {
    "flet": "flet",
    "torch": "torch",
    "vulkan": "vulkan",
    "openxr": "xr",
    "tensorrt": "tensorrt",
    "torch_directml": "torch_directml",
    "openvino": "openvino",
    "windows_capture": "windows_capture",
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
                "compute_queue_family": device.compute_queue_family_index,
                "transfer_queue_family": device.transfer_queue_family_index,
                "timeline_semaphore_enabled": device.timeline_semaphore_enabled,
                "adapter_luid": int(getattr(device, "adapter_luid", 0)),
                "adapter_identity": getattr(device, "adapter_identity", None),
                "vendor_id": int(getattr(device, "vendor_id", 0)),
                "device_id": int(getattr(device, "device_id", 0)),
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


def _probe_gpu_producers() -> dict[str, object]:
    """Report producer selection without loading vendor interop handles."""
    override = os.environ.get("D2S_VULKAN_GPU_BACKEND")
    report: dict[str, object] = {
        "selection": "auto",
        "override": override.strip() if override and override.strip() else None,
        "selected_backend": "none",
        "cuda": {"torch_available": False, "available": False},
        "rocm": {"torch_available": False, "hip_version": None, "available": False},
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        report["cuda"] = {
            "torch_available": True,
            "available": cuda_available and not bool(hip_version),
        }
        report["rocm"] = {
            "torch_available": True,
            "hip_version": str(hip_version) if hip_version else None,
            "available": bool(hip_version) and cuda_available,
        }
        if override and override.strip().lower() not in {"", "auto", "default"}:
            report["selection"] = "override"
            report["selected_backend"] = override.strip().lower()
        elif bool(hip_version) and cuda_available:
            report["selected_backend"] = "rocm"
        elif cuda_available:
            report["selected_backend"] = "cuda"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def hardware_regression_matrix() -> list[dict[str, object]]:
    """Return explicit hardware validation entries without claiming support."""
    common = {
        "test_entry": (
            "src/python3/python.exe -m pytest tests/test_backend_capabilities.py "
            "tests/test_windows_capture_event.py "
            "tests/test_openvino_native_depth.py tests/test_compute_backend.py -q"
        ),
        "runtime_log_prefix": "[D2S_BACKEND_STATUS]",
        "diagnostic_log_keys": [
            "depth_backend",
            "stereo_backend",
            "fallback_reasons",
            "capture_adapter_luid",
            "capture_adapter_uuid",
            "capture_pci_bdf",
            "resource_kind",
            "resource_format",
            "directml_resource_mode",
            "gpu_to_cpu",
            "gpu_copy_count",
            "zero_copy",
        ],
        "status": "待硬件验证",
    }
    entries = [
        ("Windows", "NVIDIA", "TensorRT -> PyTorch CUDA -> DirectML -> CPU"),
        ("Windows", "AMD", "MIGraphX/ROCm -> PyTorch ROCm -> DirectML -> CPU"),
        ("Windows", "Intel", "OpenVINO -> PyTorch XPU -> DirectML -> CPU"),
        ("Windows", "其他 DX12/DirectML（含国产 GPU）", "DirectML -> CPU"),
        ("Linux", "NVIDIA", "TensorRT -> PyTorch CUDA -> CPU"),
        ("Linux", "AMD", "MIGraphX/ROCm -> PyTorch ROCm -> CPU"),
        ("Linux", "Intel/其他 GPU", "已验证厂商后端 -> CPU"),
        ("macOS", "Apple Silicon", "本项目不实现原生 CoreML/MPSGraph 链路"),
        ("macOS", "Intel Mac", "本项目不实现原生 CoreML/Metal 链路"),
        ("跨平台", "GPU 后端不可用", "CPU 推理 -> CPU 合成/输出"),
    ]
    return [
        {
            **common,
            "os": system,
            "hardware": hardware,
            "expected_inference_order": inference_order,
            "expected": (
                "成功时输出实际后端与资源身份；失败时明确回退原因，"
                "不得静默声称 zero_copy"
            ),
        }
        for system, hardware, inference_order in entries
    ]


def build_capability_report() -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    filament_platforms = {
        "win32": ("windows", "filament_bridge.dll"),
        "darwin": ("macos", "libfilament_bridge.dylib"),
        "linux": ("linux", "libfilament_bridge.so"),
    }
    filament_platform = filament_platforms.get(sys.platform)
    filament_path = (
        src_root / "xr_viewer" / "native" / filament_platform[0]
        / filament_platform[1]
        if filament_platform else None
    )
    try:
        from capture.capabilities import probe_capture_capabilities

        capture_report = probe_capture_capabilities()
    except Exception as exc:
        capture_report = {
            "status": "probe_failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
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
        "capture": capture_report,
        "vulkan": _probe_vulkan_device(),
        "gpu_producers": _probe_gpu_producers(),
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
