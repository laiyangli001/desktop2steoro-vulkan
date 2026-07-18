import platform


OS_NAME = platform.system()


def resolve_capture_tool(raw_value, os_name=OS_NAME):
    """Pick the OS- and device-specific capture backend when settings use auto/none."""
    if raw_value and raw_value != "none":
        return raw_value
    if os_name == "Windows":
        try:
            import torch

            if torch.cuda.is_available():
                if getattr(torch.version, "hip", None) is not None:
                    return "WindowsCaptureROCm"
                return "WindowsCaptureCUDA"
        except Exception:
            pass
        try:
            import torch_directml

            if torch_directml.is_available() and torch_directml.device_count() > 0:
                return "DXCamera"
        except Exception:
            pass
        return "DXCamera"
    if os_name == "Darwin":
        return "ScreenCaptureKit"
    return "DXCamera"


_resolve_capture_tool = resolve_capture_tool
